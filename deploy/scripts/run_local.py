#!/usr/bin/env python3
"""Локальный раннер валидации (замена SSH-оркестрации run_remote.py).

Берёт EVAL_CONFIGS[name] из evaluate.py, накладывает абсолютные локальные пути
из configs/local_paths.yaml и вызывает cfg["eval_fn"](cfg). Результаты пишутся
в репозиторные results/<model>/ и plots/ (через симлинки deploy/results, deploy/plots).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]        # /home/mk/CarCV-metrics
EVAL_DIR = REPO_ROOT / "deploy" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))
from evaluate import EVAL_CONFIGS  # noqa: E402

DEFAULT_PATHS = REPO_ROOT / "configs" / "local_paths.yaml"
OVERLAY_KEYS = ("model_path", "labels_path", "data_dir", "results_dir",
                "conf_thr", "eval_classes")


def preload_cuda() -> None:
    """Подгрузить CUDA/cuDNN из nvidia-*-cu12 pip-пакетов venv до создания ORT-сессий.

    Без этого onnxruntime не находит libcublasLt.so.12 и тихо откатывается на
    CPUExecutionProvider. На машинах без GPU/пакетов — мягкое предупреждение, не фатально.
    """
    try:
        import onnxruntime as ort
        ort.preload_dlls()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] onnxruntime.preload_dlls() не сработал ({e}); возможен CPU-фолбэк")


def load_paths(path: Path) -> dict:
    """configs/local_paths.yaml → dict (пусто, если файла нет)."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def overlay_config(base_cfg: dict, overlay: dict) -> dict:
    """Копия base_cfg с наложенными разрешёнными ключами overlay (eval_fn неприкосновенен)."""
    cfg = dict(base_cfg)
    for k, v in (overlay or {}).items():
        if k in OVERLAY_KEYS:
            cfg[k] = v
    return cfg


def select_models(names: list[str], configs: dict) -> list[str]:
    if "all" in names:
        return list(configs.keys())
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CARS local evaluation runner")
    p.add_argument("--models", nargs="+",
                   choices=list(EVAL_CONFIGS.keys()) + ["all"], default=["all"])
    p.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    args = p.parse_args(argv)

    preload_cuda()   # активировать CUDAExecutionProvider до первой ORT-сессии
    paths = load_paths(args.paths)
    for name in select_models(args.models, EVAL_CONFIGS):
        cfg = overlay_config(EVAL_CONFIGS[name], paths.get(name, {}))
        print(f"─── local eval: {name} ───")
        result = cfg["eval_fn"](cfg)
        if "error" in result:
            # Ошибки/UNDEF отдельной модели — валидный результат кампании (НЕ фатально):
            # детали пишутся в results/<model>/metrics.json, exit-code остаётся 0.
            print(f"{name}: ERROR/UNDEF → {result['error']}")
        else:
            print(f"{name}: {result.get('thresholds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
