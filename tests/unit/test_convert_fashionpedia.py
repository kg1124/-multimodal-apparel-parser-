from apparel_parser.data.convert_fashionpedia import segmentation_to_yolo_lines, build_category_mapping


def test_segmentation_to_yolo_lines_normalizes_coords():
    segmentation = [[0, 0, 100, 0, 100, 100, 0, 100]]
    lines = segmentation_to_yolo_lines(segmentation, target_class=5, img_w=200, img_h=200)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "5"
    coords = [float(v) for v in parts[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)


def test_build_category_mapping_matches_by_name():
    categories = [
        {"id": 32, "name": "shoe"},
        {"id": 33, "name": "bag, wallet"},
        {"id": 0, "name": "shirt, blouse"},  # 不在映射表里，应该被忽略
    ]
    mapping = build_category_mapping(categories)
    assert mapping == {32: 5, 33: 6}
