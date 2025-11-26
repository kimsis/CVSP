import argparse
from pathlib import Path
from typing import List, Tuple
import cv2

import matplotlib.pyplot as plt
from numpy import ndarray

def parse_annotations(path: Path, format: str) -> List[Tuple[int, int, int, int]]:
    bboxes = []
    with open(path, "r") as f:
        lineno = 0
        for line in f:
            cleaned = line.strip().rstrip(',')   # removes trailing comma(s)
            parts = cleaned.split(',')
            if (format == "image"):
                if len(parts) != 8:
                    raise ValueError(f"Expected 8 fields, got {len(parts)}")
                x, y, w, h, score, category, truncated, occluded = map(int, cleaned.split(','))
            else:
                if len(parts) != 10:
                    raise ValueError(f"Expected 9 fields, got {len(parts)}")
                frame, target, x, y, w, h, score, category, truncated, occluded = map(int, cleaned.split(','))

            if category not in [1, 2] or frame not in [1]:  # Only consider pedestrian and people in frame 1
                continue
            bboxes.append(calculate_bbox(x, y, w, h))
    
    if not bboxes:
        raise ValueError(f"Could not parse annotation file: {path} at line {lineno}")
    
    return bboxes

def parse_labels(path: Path, img_width: int, img_height: int) -> List[Tuple[int, int, int, int]]:
    bboxes = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            cleaned = line.strip().rstrip(',')   # removes trailing comma(s)
            if not cleaned: # Skip empty lines
                continue

            parts = cleaned.split(' ')
            if len(parts) != 5:
                raise ValueError(f"Expected 5 fields, got {len(parts)}")

            try:
                category = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # Convert from normalized to absolute coordinates
                w_px = w * img_width
                h_px = h * img_height
                x = x_center * img_width - w_px / 2
                y = y_center * img_height - h_px / 2

                bboxes.append(calculate_bbox(x, y, w_px, h_px))

            except ValueError as e:
                raise ValueError(f"Could not parse line {lineno} in {path}: {e}")
    
    if not bboxes:
        raise ValueError(f"No valid bounding boxes found in label file: {path}")
    
    return bboxes

def calculate_bbox(x:int, y:int, w:int, h:int) -> Tuple[int, int, int, int]:
    x1 = x + w
    y1 = y + h
    return int(round(x)), int(round(y)), int(round(x1)), int(round(y1))

def draw_bbox_on_image(img: ndarray[any, any], bboxes: List[Tuple[int, int, int, int]]):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red in BGR
    
    return img

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="Path to dataset root (e.g. /path/to/dataset)")
    p.add_argument("--image", type=str, required=True, help="Image name")
    p.add_argument("--format", type=str, default="image", choices=["image", "video"], help="Source format: image or video")
    args = p.parse_args()


    dataset_root = Path('datasets') / args.dataset

    # handle image name whether or not it includes an extension
    image_name = Path(args.image)
    if format == "image":
        image_path = dataset_root / "images" / image_name.name
    else:
        image_path = dataset_root / "images" / image_name / "0000001.jpg"

    # ann_path = dataset_root / "labels" / (image_name.stem + ".txt")
    ann_path = dataset_root / "annotations" / (image_name.stem + ".txt")

    # Read image using OpenCV (returns BGR format)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # width_img, height_img = img.shape[1], img.shape[0]
    # bboxes = parse_labels(ann_path, width_img, height_img)
    bboxes = parse_annotations(ann_path, args.format)

    img_with_box = draw_bbox_on_image(img, bboxes)

    cv2.imshow(f"{image_path.name}", img_with_box)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()