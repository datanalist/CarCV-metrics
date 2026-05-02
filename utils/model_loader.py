import os
import shutil
import subprocess
from pathlib import Path
import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class TrafficCamNetLoader:
    """Load and manage ONNX TrafficCamNet model inference."""

    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Args:
            model_path: Path to ONNX model file
            providers: onnxruntime execution providers.
                      Default: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        """
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        ort.preload_dlls()
        self.session = ort.InferenceSession(model_path, providers=providers)
        active_providers = self.session.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                f"GPU inference not active. Active providers: {active_providers}\n"
                "Fix: run `uv sync` to install nvidia-cublas-cu12, "
                "nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, "
                "nvidia-curand-cu12, nvidia-cufft-cu12."
            )
        logger.info(f"GPU confirmed. Active providers: {active_providers}")
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Input: {self.input_name}, shape: {self.input_shape}")
        logger.info(f"Outputs: {self.output_names}")

    def infer(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on preprocessed image batch.

        Args:
            image_data: Numpy array, shape (N, 1, H, W) or (N, 3, H, W)

        Returns:
            Dictionary mapping output names to arrays
        """
        outputs = self.session.run(self.output_names, {self.input_name: image_data})
        return {name: out for name, out in zip(self.output_names, outputs)}


class TLTModelLoader(TrafficCamNetLoader):
    """
    Load TrafficCamNet from a TAO Toolkit (.tlt) checkpoint.

    .tlt is an encrypted Keras checkpoint produced by NVIDIA TAO/TLT. ONNX Runtime
    cannot consume it directly, so this loader transparently exports the .tlt file
    to ONNX once and then delegates inference to TrafficCamNetLoader.

    Export tools probed (in order):
      1. `tao model detectnet_v2 export` (TAO 5.x)
      2. `tao detectnet_v2 export`       (TAO 4.x)
      3. `tlt-export` / `detectnet_v2 export` (legacy TLT 3.x)

    The exported ONNX is cached at ``onnx_cache_path`` so repeated runs do not
    re-export. The model key (``tlt_encode`` for the public TrafficCamNet
    ``unpruned_v1.0``) can be supplied via argument or the ``TLT_MODEL_KEY`` /
    ``NGC_MODEL_KEY`` environment variables.
    """

    DEFAULT_KEY = "tlt_encode"

    def __init__(
        self,
        tlt_path: str,
        onnx_cache_path: Optional[str] = None,
        model_key: Optional[str] = None,
        input_dims: Tuple[int, int, int] = (3, 544, 960),
        export_command: Optional[List[str]] = None,
        force_reexport: bool = False,
        providers: List[str] = None,
    ):
        """
        Args:
            tlt_path: Path to .tlt checkpoint
            onnx_cache_path: Where to store the exported ONNX. Defaults to the
                same directory as ``tlt_path`` with the ``.onnx`` suffix.
            model_key: Decryption key. Defaults to env ``TLT_MODEL_KEY`` /
                ``NGC_MODEL_KEY``, then to ``"tlt_encode"`` (NGC public key for
                TrafficCamNet ``unpruned_v1.0``).
            input_dims: (C, H, W) tuple — required by detectnet_v2 export to
                build the ONNX with the right input shape (default 3×544×960).
            export_command: Override auto-detected export CLI. Use ``{tlt}``,
                ``{onnx}``, ``{key}``, ``{c}``, ``{h}``, ``{w}`` placeholders.
            force_reexport: If True, re-export ONNX even if cache exists.
            providers: onnxruntime execution providers (passed to parent).
        """
        tlt_path = str(Path(tlt_path).expanduser().resolve())
        if not os.path.exists(tlt_path):
            raise FileNotFoundError(f".tlt model not found: {tlt_path}")
        if not tlt_path.lower().endswith(".tlt"):
            logger.warning(
                f"Expected .tlt extension, got {tlt_path!r} — proceeding anyway."
            )

        if onnx_cache_path is None:
            onnx_cache_path = str(Path(tlt_path).with_suffix(".onnx"))
        onnx_cache_path = str(Path(onnx_cache_path).expanduser().resolve())

        if model_key is None:
            model_key = (
                os.environ.get("TLT_MODEL_KEY")
                or os.environ.get("NGC_MODEL_KEY")
                or self.DEFAULT_KEY
            )

        self.tlt_path = tlt_path
        self.onnx_cache_path = onnx_cache_path
        self.model_key = model_key
        self.input_dims = input_dims

        if force_reexport or not os.path.exists(onnx_cache_path):
            self._export_tlt_to_onnx(export_command)
        else:
            logger.info(f"Using cached ONNX export: {onnx_cache_path}")

        super().__init__(model_path=onnx_cache_path, providers=providers)

    def _export_tlt_to_onnx(self, override_command: Optional[List[str]]) -> None:
        """Export ``self.tlt_path`` → ``self.onnx_cache_path`` via TAO/TLT CLI."""
        c, h, w = self.input_dims
        Path(self.onnx_cache_path).parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect available export CLI if no override provided.
        if override_command is None:
            command_templates = self._candidate_export_commands()
        else:
            command_templates = [override_command]

        last_error: Optional[Exception] = None
        for tmpl in command_templates:
            cmd = [
                arg.format(
                    tlt=self.tlt_path,
                    onnx=self.onnx_cache_path,
                    key=self.model_key,
                    c=c,
                    h=h,
                    w=w,
                )
                for arg in tmpl
            ]
            executable = cmd[0]
            if shutil.which(executable) is None:
                logger.debug(f"Skipping export candidate, not on PATH: {executable}")
                continue

            logger.info(f"Exporting .tlt → ONNX via: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                if not os.path.exists(self.onnx_cache_path):
                    raise RuntimeError(
                        f"Export command finished but ONNX file is missing: "
                        f"{self.onnx_cache_path}"
                    )
                logger.info(f"ONNX exported successfully → {self.onnx_cache_path}")
                return
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                last_error = exc
                logger.warning(f"Export attempt failed ({executable}): {exc}")

        raise RuntimeError(
            "Failed to export .tlt → .onnx. None of the TAO/TLT export CLIs "
            "succeeded. Install NVIDIA TAO Toolkit "
            "(`pip install nvidia-tao`) or TLT 3.x, or pre-export the model "
            "manually with:\n"
            "  tao model detectnet_v2 export -m <model>.tlt -k <key> "
            "--export_format onnx -o <model>.onnx\n"
            f"Last error: {last_error}"
        )

    @staticmethod
    def _candidate_export_commands() -> List[List[str]]:
        """Ordered list of CLI templates for .tlt → .onnx export."""
        return [
            # TAO 5.x
            [
                "tao", "model", "detectnet_v2", "export",
                "-m", "{tlt}",
                "-k", "{key}",
                "--export_format", "onnx",
                "-o", "{onnx}",
                "--input_dims", "{c},{h},{w}",
            ],
            # TAO 4.x
            [
                "tao", "detectnet_v2", "export",
                "-m", "{tlt}",
                "-k", "{key}",
                "--export_format", "onnx",
                "-o", "{onnx}",
                "--input_dims", "{c},{h},{w}",
            ],
            # Legacy TLT 3.x via tlt-export
            [
                "tlt-export", "detectnet_v2",
                "-m", "{tlt}",
                "-k", "{key}",
                "--export_format", "onnx",
                "-o", "{onnx}",
                "--input_dims", "{c},{h},{w}",
            ],
        ]
