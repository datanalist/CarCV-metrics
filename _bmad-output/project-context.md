---
project_name: CarCV-metrics
user_name: Vallo
date: 2026-05-16
sections_completed:
  ['technology_stack', 'language_rules', 'framework_rules', 'testing_rules', 'quality_rules', 'workflow_rules', 'anti_patterns']
status: complete
rule_count: 78
optimized_for_llm: true
existing_patterns_found: 12
---

# Project Context for AI Agents

_Этот файл содержит критические правила и паттерны, которым AI-агенты должны следовать при реализации кода в этом проекте. Фокус — на неочевидных деталях, которые агенты могут упустить._

---

## Technology Stack & Versions

**Runtime:** Python ≥3.10, venv in `./venv`, package manager **uv** (NEVER pip).

**ML Inference:**
- `onnxruntime-gpu` ≥1.24.3 (local); pinned to `==1.20.1` in `deploy/requirements.txt` — version divergence is intentional
- `torch` ≥2.11.0 + `torchvision` ≥0.26.0 from custom index `pytorch-cu130` (CUDA 13.0) — **local development only**
- **Remote eval servers**: NVIDIA drivers and PyTorch are **pre-installed** — agents must NOT add `torch`/`torchvision`/`nvidia-*` to `deploy/requirements.txt` or to `runtime.remote_with_packages` in hydra configs

**CV/Data:** `opencv-python-headless` ≥4.8 (NEVER `opencv-python`), `numpy` ≥1.24, `pillow` ≥12.1, `pyarrow` ≥22, `pandas` ≥2.3, `polars` ≥1.40.

**Configs/Remote:** `hydra-core` ≥1.3.2 (configs under `configs/experiment/*.yaml`), `paramiko` ≥4.0 (SSH bundle to remote), `python-Levenshtein` ≥0.21 (OCR edit distance).

**Deploy-only (not in pyproject.toml):** `datasets`, `huggingface_hub`, `kaggle`, `pycocotools` — declared in `deploy/requirements.txt`.

**Hardware contract:**
- **Production target:** NVIDIA Jetson Orin Nano 8GB (JetPack 5.x, CUDA 11.4+, TensorRT 8.5+, DeepStream SDK 6.2+) — models run as `*.engine` (TensorRT FP16).
- **Evaluation:** any remote GPU server with NVIDIA driver + PyTorch already installed. Server count and GPU model are variable — do NOT hardcode specific hosts (e.g. `qudata2`, `qudata5`) into application code. Treat `runtime.remote_host` in hydra YAML as the single source of truth.

**Version notes for agents:**
- `pyproject.toml` (local) and `deploy/requirements.txt` (remote) are intentionally out of sync. Do NOT "fix" them by aligning versions.
- Custom PyTorch index `pytorch-cu130` is declared via `[[tool.uv.index]]`; `pip install torch` will fetch the wrong CUDA build. Always `uv add` / `uv sync` locally.
- Before remote run, verify environment with `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"` — do NOT reinstall.

## Critical Implementation Rules

### Language-Specific Rules (Python)

**Paths:**
- Use `pathlib.Path` exclusively. `ROOT = Path(__file__).parent.parent` is the canonical project root inside `deploy/evaluation/`. Compose with `ROOT / "subdir" / "file"` — NEVER `os.path.join`, NEVER hardcoded `/home/...` paths.
- Hydra configs reference paths relative to the project root; do not resolve them with `os.getcwd()`.

**ONNX Runtime:**
- Always create session with providers `["CUDAExecutionProvider", "CPUExecutionProvider"]` (in that order) so CPU is fallback, not failure.
- Log the active provider after session creation: `sess.get_providers()[0]`.
- Input tensors are **NCHW float32** for all models except `LPR_STN_PRE_POST` which is **NHWC uint8** (no normalization). Confirm with `sess.get_inputs()[0].shape` before guessing.

**Preprocessing dispatch (must not be mixed up):**

| Model family | Color | Scaling | Shape |
|---|---|---|---|
| Generic ImageNet classifier (`bae_model_f3`) | BGR→RGB | `/255` then `(x-mean)/std` | NCHW |
| TAO classifier (`VehicleMakeNet`, `VehicleTypeNet`) | **BGR (no swap)** | `x - offsets`, no scale | NCHW |
| DetectNet_v2 (`TrafficCamNet`, `LPDNet`) | BGR→RGB | `/255` | NCHW |
| LPR (`LPR_STN_PRE_POST`) | BGR→RGB | **uint8, no normalization** | NHWC |

Per-model offsets are in `configs/experiment/*.yaml` — do NOT hardcode in Python. VehicleMakeNet uses `(104, 117, 124)` BGR offsets, generic TAO default is `(103.939, 116.779, 123.68)`.

**OpenCV:**
- `cv2.imread` returns BGR uint8 by default. Always check for `None` (corrupt/missing file) and `continue` rather than raise.
- `cv2.resize` order is `(width, height)`, not `(height, width)`.

**NumPy:**
- Use `[None]` to add batch dim, `transpose(2, 0, 1)` for HWC→CHW. Don't reach for `einops` — not in dependencies.
- Softmax: subtract max before exp for numerical stability — see `softmax()` in `deploy/evaluation/evaluate.py`.

**Logging:**
- One logger per module via `logging.getLogger(__name__)`. The root handler in `evaluate.py` writes to BOTH stderr and `logs/eval_{os.uname().nodename}.log` — never override with `print`/`sys.stderr.write`.
- The hostname-in-filename pattern is load-bearing: parallel runs on multiple servers must not collide.

**JSON/CSV output:**
- Write JSON via `Path.write_text(json.dumps(obj, indent=2))` — preserves pathlib usage and forces UTF-8.
- For per-class CSV use pandas with `index=False`. Polars preferred for large frames, pandas for small/legacy IO.

**Imports inside `deploy/evaluation/`:**
- Flat layout — `from metrics import ...`, NOT `from deploy.evaluation.metrics import ...`. The bundle runs from inside this folder on remote.

**Type hints & comments:**
- Light annotations only (`-> dict`, `-> np.ndarray`). No pydantic, no dataclasses unless data is reused across files.
- Comments only for WHY (e.g. "VehicleMakeNet expects BGR offsets (104,117,124), not ImageNet"). Never describe WHAT — naming should carry that.

**Labels file quirk:**
- `load_labels()` accepts both newline- and **semicolon-separated** label files (NGC ships `;`-joined `labels.txt`). Don't "simplify" to only one format.

### Framework-Specific Rules

#### Hydra (config-driven experiments)

- Every evaluation experiment is defined by a single YAML in `configs/experiment/`. The canonical block layout is **fixed**: `experiment`, `runtime`, `model`, `data`, `evaluation`, `artifacts` — add new blocks only with strong justification.
- All paths inside YAML are relative to project root and joined via `ROOT / cfg["..."]` in Python — never `os.path.expanduser` inside config values.
- Model-family preprocessing constants (`net_scale_factor`, `mean_*`, `offsets`, `output_*_name`) live in YAML, NOT in Python. If a new model needs a new constant, add to YAML and read with `cfg.get("key", default)`.
- Hydra config groups are not used yet — single flat YAML per experiment. Don't introduce `_target_` / instantiate patterns without buy-in.
- Class mapping between dataset taxonomy and model output classes (`data.category_map`) is the seam where domain gaps are absorbed. Always extend this map rather than patching Python.

#### ONNX Runtime

- One `InferenceSession` per model — do NOT share sessions across model families even if providers match.
- Read input/output names from the session (`sess.get_inputs()[0].name`, `sess.get_outputs()[i].name`) rather than hardcoding, EXCEPT for DetectNet_v2 where the names (`output_cov/Sigmoid`, `output_bbox/BiasAdd`) are in the YAML config and must match.
- Sessions are not thread-safe; if you need parallelism, instantiate per-thread.
- Batch dimension is fixed to 1 in current code paths — batching support is opt-in per model and requires changing the YAML `evaluation.batch_size` AND verifying the model supports dynamic batch.

#### Paramiko / SSH remote execution (canonical pattern)

- Bundling protocol:
  1. Stage code + config in a tmp tarball locally.
  2. SCP to `runtime.remote_bundle_root` (default `/dev/shm/<exp_name>` — tmpfs, fast, ephemeral).
  3. SSH-execute `uv run` (or `python`) on the remote, capturing stdout/stderr.
  4. SCP `outputs/` directory back to `artifacts.local_output_root/raw_remote/`.
  5. If `runtime.cleanup_remote_bundle: true` — `rm -rf` the bundle dir.
- `runtime.remote_with_packages` lists ONLY experiment-specific deps (`onnxruntime-gpu`, `opencv-python-headless`, …). Never include `torch`, `nvidia-*`, or system libs — those are pre-existing on remotes.
- The legacy `deploy/scripts/deploy_to_servers.sh` uses rsync + hardcoded hosts; treat it as deprecated reference, not the source of truth. New work goes through hydra `runtime` block.
- Remote host names come from `~/.ssh/config` — application code reads only `runtime.remote_host`.

#### DeepStream / TensorRT (production target, read-only for this repo)

- Production runs on Jetson with TensorRT `*.engine` files; this metrics repo does NOT regenerate engines. We evaluate the matching ONNX exports on x86 servers and treat numbers as the upper bound for engine accuracy.
- DeepStream PGIE/SGIE configs live in `configs/dstest2_*.txt` (INI-style for nvinfer). Preprocessing keys there (`net-scale-factor`, `offsets`, `model-color-format`) are the source-of-truth that hydra YAMLs must mirror — divergence between them is a bug.
- `model-color-format=0` means RGB, `=1` means BGR. Easy to invert.

#### Jupyter / Notebooks

- Reproducible notebooks live under `notebooks/{section}_{topic}.ipynb` and consume the SAME metrics JSON written by `evaluate.py` (no re-running inference inside the notebook unless explicitly an exploration cell).
- Figures saved to `plots/{model_name}_*.png` AND `notebooks/<exp>/images/` for embedding — keep both paths in sync via the YAML `artifacts.local_figure_root`.

### Testing Rules

- No formal test suite (`pytest`, `unittest`) is in place yet — this is an evaluation/metrics project where the **dataset is the test fixture**. Validation comes from per-model metric thresholds defined in `evaluation.target_*` YAML keys.
- `check_thresholds()` in `metrics.py` is the pass/fail gate; agents adding new models MUST define thresholds in YAML and call this helper — never inline numeric comparisons.
- For utility code (preprocessing, decoders, metric helpers) add `pytest` tests under `tests/` mirroring the module path. Targets: pure functions with deterministic outputs (NMS, CTC decode, label normalization, IoU).
- Tests must NOT require GPU, network, or real model files. Use small synthetic tensors and tiny fixture images checked into `tests/fixtures/` (≤100 KB each).
- For OCR-like decoders, include adversarial fixtures (empty input, all-blank, repeated chars, label-length mismatch).
- When measuring inference timing, run a **warmup pass** before the timed loop and report **median + p95**, not mean (CUDA tail latency).
- Out-of-distribution samples must be **counted and logged**, not silently dropped — see `skipped_oo_dist` counter in `eval_vehiclemakenet`.

### Code Quality & Style Rules

- No linter/formatter is configured. Default to **PEP 8** + 100-char lines. If introducing one, use **ruff** (covers black + isort + flake8), config via `[tool.ruff]` in `pyproject.toml`.
- Imports grouped stdlib → third-party → local, separated by blank lines (as in `deploy/evaluation/evaluate.py`).
- Section banner comments (`# ─── ModelName ──────`) separate model evaluators inside `evaluate.py`. Keep this style; do not switch to docstring-only or numbered headings.
- Function naming: `eval_{model_name}` for per-model evaluators, `preprocess_{family}` for preprocessing, `*_decode` for output decoders. Stick to these prefixes.
- File naming: `snake_case.py`. Notebooks use `{section_number}_{Title_PascalCase}.ipynb` (e.g. `3.6_LPR_STN_PRE_POST_Baseline_Evaluation.ipynb`).
- Constants in `UPPER_SNAKE_CASE` at module top (`IMAGENET_MEAN`, `NGC_MAKES`). Per-model magic numbers (offsets, strides) live in YAML, not constants.
- Docstrings only on non-obvious decoders (e.g. `detectnet_v2_decode`); skip on `def eval_x(cfg)` — the name + YAML is self-documenting.
- f-strings everywhere; no `%`-formatting or `.format()`.

### Development Workflow Rules

- **Branching:** topic branches off `main`, prefix matches BMAD conventions seen in git history: `chore/`, `feat/`, `fix/`, `docs/`, `refactor/`. Current example: `chore/bmad-deploy-artifacts`.
- **Commit messages:** Russian or English allowed, follow Conventional Commits prefix (`chore:`, `feat:`, `fix:`, `docs:`). Recent history mixes both languages — match the area you're touching (Russian for `docs/`, English for code-only).
- **Don't commit:**
  - Model weights (`*.onnx`, `*.pt`, `*.pth`, `*.engine`, `*.safetensors`) — `.gitignore` already covers
  - Datasets (`data/`, `datasets/`) — `.gitignore` already covers
  - `logs/`, `venv/`, `.env*`
  - Pre-computed result JSON/CSV unless explicitly a published baseline reference
- **Do commit:** all `configs/experiment/*.yaml`, all code under `deploy/`, all `docs/`, `notebooks/` (with outputs cleared via `nbstripout` if used).
- Large baseline results go to `results_collected/<server>/` and are committed when they represent a frozen reference (see `results_collected/qudata2/`).
- `_bmad-output/` is committed — it carries planning artifacts and now this project-context.
- **Reproducibility:** every experiment must produce: `metrics.json`, `per_class_metrics.csv` (if applicable), plots `*.png`, and an entry in `results/SUMMARY.md`. Plus a reproducible notebook under `notebooks/`.

### Critical Don't-Miss Rules

**Anti-patterns to avoid:**
- ❌ Mixing BGR/RGB preprocessing — TAO classifiers expect raw BGR with offsets, NOT ImageNet RGB. This is the #1 source of silent accuracy collapse.
- ❌ Reinstalling `torch` / NVIDIA stack on remote servers — they are pre-existing. Adding them to `runtime.remote_with_packages` will fight the environment.
- ❌ Hardcoding remote hostnames in Python (`qudata2`, `qudata5`). Server topology is variable; read from hydra YAML `runtime.remote_host`.
- ❌ Hardcoding dataset/model absolute paths (`/home/mk/Загрузки/...`) in code. Such paths only belong in user-local YAML overrides.
- ❌ Calling `pip install` inside this repo. Always `uv add` / `uv sync`.
- ❌ Using `opencv-python` (with GUI) instead of `opencv-python-headless` — breaks headless remote servers.
- ❌ Skipping the CPU fallback provider — pure `["CUDAExecutionProvider"]` lists hard-fail on CPU-only debug runs.
- ❌ Comparing `cv2.imread` result against falsy without explicit `is None` — numpy arrays raise on truth-value check.
- ❌ Aligning `pyproject.toml` and `deploy/requirements.txt` "for consistency" — they target different stacks intentionally.
- ❌ "Fixing" the BMM config to English. Project artifacts stay in Russian (see [Artifact language memory]).

**Domain-specific edge cases:**
- Out-of-distribution labels: not every dataset sample maps onto model classes (e.g. mad-cars has brands outside NGC's 20). Skip + log + count, then exclude from accuracy denominator.
- US-trained LPR/LPDNet on RU plates: known domain gap. Document in `SUMMARY.md` as a separate section, never silently treat the numbers as production-ready.
- LPR CTC decoding uses `-` as a class separator that must be stripped post-decode along with blanks.
- DetectNet_v2 outputs are GRID-relative; you must apply `stride=16`, `bbox_norm=35.0`, and scale-back to original image coords before computing IoU against ground truth.
- BDD100K's bbox key is `box2d` in some versions, `bbox2d` in others — read both (see `eval_trafficcamnet`).

**Security / safety:**
- `.env` files are gitignored — never commit credentials. Kaggle/HF tokens go in `~/.config/kaggle/` and `~/.cache/huggingface/`, not in repo.
- Remote bundles in `/dev/shm/` are world-readable on shared hosts; do not place secrets in the bundle. SSH keys stay on local machine.
- SSH config lives in `~/.ssh/config`; agents must never write SSH host configs from code.

**Performance gotchas:**
- ONNX Runtime first inference includes graph optimization — always **warm up** before timing.
- `cv2.imread` is single-threaded; if dataset loading bottlenecks, use `concurrent.futures.ThreadPoolExecutor` with `cv2.IMREAD_REDUCED_*` flags, NOT multiprocessing (memory blow-up with full-size images).
- pandas with `iterrows()` on >10k rows is a trap. Use `polars` or vectorized ops.
- TQDM updates inside tight CUDA loops add measurable overhead; set `mininterval=1.0` for inference progress bars.

---

## Usage Guidelines

**For AI agents:**
- Read this file before implementing any code in `CarCV-metrics`.
- Follow ALL rules exactly as documented; when in doubt prefer the stricter option.
- All written artifacts (PRDs, research docs, dataset docs, story specs, summaries) in **Russian**, despite the English BMM config — this is an explicit project convention. Code identifiers, filenames, log messages: English/ASCII.
- Update this file when new patterns emerge or rules become stale.

**For humans:**
- Keep this file lean — every bullet must add information not derivable from reading the code.
- Update when technology stack changes, when a new model family is added, or when remote infra topology shifts.
- Review periodically and prune rules that become self-evident or outdated.

Last Updated: 2026-05-16
