import json
import random
from pathlib import Path

import cv2

annos_dir = Path("/root/autodl-tmp/project/deepfashion2.zip/train/annos")
image_dir = Path("/root/autodl-tmp/project/deepfashion2.zip/train/image")
out_dir = Path("/root/autodl-tmp/project/landmark_top_samples")
out_dir.mkdir(exist_ok=True)

json_files = list(annos_dir.glob("*.json"))
random.seed(7)
random.shuffle(json_files)

count = 0
NEED = 6
scanned = 0
top_seen = 0

for json_path in json_files:
    if count >= NEED:
        break
    scanned += 1
    with open(json_path) as f:
        data = json.load(f)
    for key, item in data.items():
        if not key.startswith("item"):
            continue
        if item.get("category_name") != "short sleeve top":
            continue
        top_seen += 1
        if item.get("viewpoint") != 2:
            continue
        if item.get("zoom_in") != 1:
            continue
        if item.get("occlusion", 3) > 2:
            continue

        image_path = image_dir / (json_path.stem + ".jpg")
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        landmarks = item.get("landmarks", [])
        for i in range(0, len(landmarks), 3):
            x, y, v = landmarks[i], landmarks[i+1], landmarks[i+2]
            idx = i // 3
            if v == 0:
                continue
            color = (0, 255, 0) if v == 2 else (0, 165, 255)
            cv2.circle(img, (int(x), int(y)), 5, color, -1)
            cv2.putText(img, str(idx), (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out_path = out_dir / f"top_{count}_{json_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)
        count += 1
        break

print(f"扫描了{scanned}个文件，看到{top_seen}个short_sleeve_top，保存了{count}张")
