import os

import cv2
import numpy as np
from pycocotools.coco import COCO

ROOT_DIR = "fire_dataset"


def convert_split(split_dir):
    json_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        print(f"JSON not found in {split_dir}")
        return

    coco = COCO(json_path)
    os.makedirs(os.path.join(split_dir, "masks"), exist_ok=True)

    print(f"Processing {split_dir} ...")

    for img_id in coco.imgs:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        h, w = img_info["height"], img_info["width"]

        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = ann["category_id"]

            m = coco.annToMask(ann)
            mask[m > 0] = cat_id

        out_path = os.path.join(
            split_dir,
            "masks",
            file_name.replace(".jpg", ".png").replace(".jpeg", ".png"),
        )
        cv2.imwrite(out_path, mask)

    print(f"Done {split_dir}")


for sub in ["train", "valid", "test"]:
    sub_dir = os.path.join(ROOT_DIR, sub)
    if os.path.isdir(sub_dir):
        convert_split(sub_dir)
