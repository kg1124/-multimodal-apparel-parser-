import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from apparel_parser.data.convert_deepfashion2 import convert_one_file
from apparel_parser.common.constants import TARGET_CLASSES

# 换成你确认过是"多polygon裤子/外套"的一张图片文件名
json_path = Path("/root/autodl-tmp/project/deepfashion2.zip/train/annos/000042.json")
image_path = Path("/root/autodl-tmp/project/deepfashion2.zip/train/image/000042.jpg")

lines = convert_one_file(json_path, image_path)
print(f"这张图生成了 {len(lines)} 行标注")
for line in lines:
    parts = line.split()
    cls = int(parts[0])
    n_points = (len(parts) - 1) // 2
    print(f"  类别={TARGET_CLASSES[cls]}, 点数={n_points}")

# 把合并后的轮廓画出来存成图片，肉眼确认是不是一个完整的裤子形状
img = cv2.imread(str(image_path))
h, w = img.shape[:2]
for line in lines:
    parts = line.split()
    coords = [float(x) for x in parts[1:]]
    pts = []
    for i in range(0, len(coords), 2):
        pts.append([int(coords[i] * w), int(coords[i + 1] * h)])
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 2)

out_path = "/root/autodl-tmp/project/sanity_check.jpg"
cv2.imwrite(out_path, img)
print(f"已保存可视化结果到 {out_path}，下载下来肉眼看一下")
