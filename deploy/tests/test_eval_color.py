import numpy as np
import evaluate


def test_color_classes_alphabetical_15():
    assert evaluate.COLOR_CLASSES == [
        "beige", "black", "blue", "brown", "gold", "green", "grey", "orange",
        "pink", "purple", "red", "silver", "tan", "white", "yellow"]
    assert len(evaluate.COLOR_CLASSES) == 15


def test_hex_to_cars_color_mapping():
    h = evaluate.HEX_TO_CARS_COLOR
    assert h["000000"] == "black"
    assert h["ffffff"] == "white"
    assert h["9966cc"] == "purple"
    assert h["0088ff"] == "blue"
    assert "tan" not in h.values()


def test_preprocess_color_shape_and_norm():
    img = np.full((10, 10, 3), 127, dtype=np.uint8)   # BGR
    inp = evaluate.preprocess_color(img, size=384)
    assert inp.shape == (1, 3, 384, 384)
    assert inp.dtype == np.float32
    expected_r = ((127 / 255.0) - 0.43) / 0.27
    assert np.allclose(inp[0, 0].mean(), expected_r, atol=1e-3)


def test_load_madcars_color_index_dedups_by_car_id(tmp_path):
    jl = tmp_path / "images_index.jsonl"
    jl.write_text(
        '{"image_id":"0_0","image_path":"images/0_0.jpg","car_id":"0","color":"000000"}\n'
        '{"image_id":"0_1","image_path":"images/0_1.jpg","car_id":"0","color":"000000"}\n'
        '{"image_id":"1_0","image_path":"images/1_0.jpg","car_id":"1","color":"ffffff"}\n')
    rows = evaluate.load_madcars_color_index(jl, dedup=True)
    assert len(rows) == 2
    assert {r["car_id"] for r in rows} == {"0", "1"}
