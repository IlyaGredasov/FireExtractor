import argparse
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations import Compose
from albumentations import Normalize
from albumentations import Resize
from albumentations.pytorch import ToTensorV2

PALETTE = {
    0: (0, 0, 0),  # bg
    1: (0, 0, 255),  # fire
    2: (192, 255, 255),  # smoke
}


def overlay_mask(img_bgr: np.ndarray, mask_index: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = mask_index.shape[:2]
    color = np.zeros((h, w, 3), np.uint8)
    for class_index, bgr in PALETTE.items():
        if class_index == 0:
            continue
        color[mask_index == class_index] = bgr

    blended = cv2.addWeighted(img_bgr, 1 - alpha, color, alpha, 0)

    out = img_bgr.copy()
    m = (mask_index > 0).astype(np.uint8) * 255
    cv2.copyTo(blended, m, out)

    for class_index in np.unique(mask_index):
        if class_index == 0:
            continue
        cnt = (mask_index == class_index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="fire_girl.jpg")
    ap.add_argument("--weights", type=str, default="best_unet.pth")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--classes", type=int, default=3)  # 0=bg,1=fire,2=smoke
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3,
                     classes=args.classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    tf = Compose([
        Resize(args.img_size, args.img_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    x = tf(image=img_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_orig = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)

    base, _ = os.path.splitext(os.path.basename(args.image))
    mask_path = f"{base}_mask.png"
    cv2.imwrite(mask_path, pred_orig)

    overlay = overlay_mask(img_bgr, pred_orig, alpha=0.45)
    overlay_path = f"{base}_masked.png"
    cv2.imwrite(overlay_path, overlay)

    print(f"Saved: {mask_path}, {overlay_path}")


if __name__ == "__main__":
    main()
