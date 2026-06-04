import run_local


def test_overlay_config_only_allowed_keys():
    base = {"model_path": "models/x.onnx", "data_dir": "data/x",
            "results_dir": "results/x", "eval_fn": object()}
    overlay = {"model_path": "/abs/x.onnx", "data_dir": "/abs/data",
               "eval_fn": "EVIL", "junk": 1}
    out = run_local.overlay_config(base, overlay)
    assert out["model_path"] == "/abs/x.onnx"
    assert out["data_dir"] == "/abs/data"
    assert out["eval_fn"] is base["eval_fn"]      # eval_fn не перетирается overlay'ем
    assert "junk" not in out
    assert base["model_path"] == "models/x.onnx"  # исходник не мутирован


def test_select_models_all_expands():
    cfgs = {"a": {}, "b": {}}
    assert run_local.select_models(["all"], cfgs) == ["a", "b"]
    assert run_local.select_models(["b"], cfgs) == ["b"]


def test_load_paths_missing_returns_empty(tmp_path):
    assert run_local.load_paths(tmp_path / "nope.yaml") == {}
