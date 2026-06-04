import evaluate


def test_discover_vmmrdb_samples_filters_ood(tmp_path):
    (tmp_path / "honda_civic_2015").mkdir()
    (tmp_path / "honda_civic_2015" / "a.jpg").write_bytes(b"x")
    (tmp_path / "honda_civic_2015" / "a2.png").write_bytes(b"x")
    (tmp_path / "toyota_camry_2016").mkdir()
    (tmp_path / "toyota_camry_2016" / "b.jpg").write_bytes(b"x")
    (tmp_path / "tesla_model3_2018").mkdir()          # OOD (нет в 20 NGC)
    (tmp_path / "tesla_model3_2018" / "c.jpg").write_bytes(b"x")

    samples = evaluate.discover_vmmrdb_samples(tmp_path, evaluate.NGC_MAKES_LOWER)
    brands = sorted({b for _, b in samples})
    assert brands == ["honda", "toyota"]
    assert any(p.suffix == ".png" for p, _ in samples)
    assert len(samples) == 3
