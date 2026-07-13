import json
import random
from pathlib import Path

import cv2

annos_dir = Path("/root/autodl-tmp/project/deepfashion2.zip/train/annos")
image_dir = Path("/root/autodl-tmp/project/deepfashion2.zip/train/image")

TARGET_CATEGORY = "short_sleeve_top"

json_files = list(annos_dir.glob("*.json"))
random.seed(1)
random.shuffle(json_files)

found = None
for json_path in json_files[:3000]:
    with open(json_path) as f:
        data = json.load(f)
    for key, item in data.items():
        if not key.startswith("item"):
            continue
        if item.get("category_name") == TARGET_CATEGORY and item.get("occlusion", 3) <= 2 and item.get("viewpoint") == 2:
            found = (json_path, key, item)
            break
    if found:
        break

if not found:
    print("还是没找到，再放宽或者换类别")
else:
    json_path, key, item = found
    image_path = image_dir / (json_path.stem + ".jpg")
    print(f"找到: {json_path.name}, {key}, {item['category_name']}, occlusion={item.get('occlusion')}")

    img = cv2.imread(str(image_path))
    landmarks = item.get("landmarks", [])
    print(f"共{len(landmarks)//3}个点")

    for i in range(0, len(landmarks), 3):
        x, y, v = landmarks[i], landmarks[i+1], landmarks[i+2]
        idx = i // 3
        if v == 0:
            continue
        color = (0, 255, 0) if v == 2 else (0, 165, 255)
        cv2.circle(img, (int(x), int(y)), 4, color, -1)
        cv2.putText(img, str(idx), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = "/root/autodl-tmp/project/landmark_check_clean.jpg"
    cv2.imwrite(out_path, img)
    print(f"已保存到 {out_path}")
