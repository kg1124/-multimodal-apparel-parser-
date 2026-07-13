import json
import sys
from pathlib import Path

import cv2

json_path = Path("/root/autodl-tmp/project/deepfashion2.zip/train/annos/000001.json")
image_path = Path("/root/autodl-tmp/project/deepfashion2.zip/train/image/000001.jpg")

with open(json_path) as f:
    data = json.load(f)

img = cv2.imread(str(image_path))

for key, item in data.items():
    if not key.startswith("item"):
        continue
    landmarks = item.get("landmarks", [])
    cat_name = item.get("category_name", "")
    print(f"{key}: {cat_name}, 共{len(landmarks)//3}个点")

    for i in range(0, len(landmarks), 3):
        x, y, v = landmarks[i], landmarks[i+1], landmarks[i+2]
        idx = i // 3
        if v == 0:
            continue
        color = (0, 255, 0) if v == 2 else (0, 165, 255)
        cv2.circle(img, (int(x), int(y)), 4, color, -1)
        cv2.putText(img, str(idx), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

out_path = "/root/autodl-tmp/project/landmark_check.jpg"
cv2.imwrite(out_path, img)
print(f"已保存到 {out_path}，下载下来看点的编号顺序")
