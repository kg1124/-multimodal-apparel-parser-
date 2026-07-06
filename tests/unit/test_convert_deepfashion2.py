from apparel_parser.data.convert_deepfashion2 import annotation_to_yolo_lines


def test_maps_category_and_normalizes_coords():
    data = {
        "item1": {
            "category_id": 5,  # vest -> 上衣（索引0）
            "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]],
        }
    }
    lines = annotation_to_yolo_lines(data, img_w=200, img_h=200)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    coords = [float(v) for v in parts[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)


def test_skips_uncovered_categories():
    data = {
        "item1": {
            "category_id": 999,  # 假设是未映射类别（鞋子/包包/配饰）
            "segmentation": [[0, 0, 100, 0, 100, 100, 0, 100]],
        }
    }
    lines = annotation_to_yolo_lines(data, img_w=200, img_h=200)
    assert lines == []
