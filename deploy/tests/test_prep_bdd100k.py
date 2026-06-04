import prep_bdd100k


def test_convert_bdd_to_labels_maps_box2d_and_filters():
    bdd = [{
        "name": "img1.jpg",
        "labels": [
            {"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
            {"category": "traffic sign", "box2d": {"x1": 5, "y1": 6, "x2": 7, "y2": 8}},
            {"category": "drivable area", "poly2d": [[0, 0]]},
            {"category": "lane"},
        ],
    }]
    out = prep_bdd100k.convert_bdd_to_labels(bdd)
    assert len(out) == 1
    rec = out[0]
    assert rec["file_name"] == "img1.jpg"
    assert rec["image_id"] == "img1.jpg"
    cats = [d["category"] for d in rec["detections"]]
    assert cats == ["car", "traffic sign"]
    assert rec["detections"][0]["bbox2d"] == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}


def test_link_images_repoints_existing_symlink(tmp_path):
    import prep_bdd100k
    out = tmp_path / "out"; out.mkdir()
    old = tmp_path / "old"; old.mkdir()
    new = tmp_path / "new"; new.mkdir()
    prep_bdd100k.link_images(out, old)
    assert (out / "images").resolve() == old.resolve()
    prep_bdd100k.link_images(out, new)          # re-point, must not raise
    assert (out / "images").resolve() == new.resolve()
    # битый симлинк: цель удалена → повторный вызов не падает
    import shutil; shutil.rmtree(new)
    prep_bdd100k.link_images(out, old)
    assert (out / "images").resolve() == old.resolve()
