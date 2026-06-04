import evaluate


def test_parse_wider_gt_xywh_to_xyxy_and_attrs(tmp_path):
    txt = tmp_path / "gt.txt"
    txt.write_text(
        "0--Parade/0_Parade_1.jpg\n"
        "2\n"
        "10 20 30 40 1 0 0 0 2 0 \n"
        "5 5 0 0 0 0 0 1 0 0 \n"          # w=h=0 → отбрасывается
        "14--Traffic/14_t.jpg\n"
        "1\n"
        "0 0 8 8 0 0 0 0 0 0 \n")
    gt = evaluate.parse_wider_gt(txt)
    assert set(gt.keys()) == {"0--Parade/0_Parade_1.jpg", "14--Traffic/14_t.jpg"}
    boxes = gt["0--Parade/0_Parade_1.jpg"]["boxes"]
    assert boxes == [[10, 20, 40, 60]]    # [x1,y1,x1+w,y1+h]; нулевой бокс отброшен
    attrs = gt["0--Parade/0_Parade_1.jpg"]["attrs"]
    assert attrs[0]["blur"] == 1 and attrs[0]["invalid"] == 0 and attrs[0]["occlusion"] == 2
