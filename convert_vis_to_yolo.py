import argparse
import os
import cv2

#   0: pedestrian
#   1: people
#   2: bicycle
#   3: car
#   4: van
#   5: truck
#   6: tricycle
#   7: awning-tricycle
#   8: bus
#   9: motor

CLASS_MAP = {
    1: 0, # person
    2: 0, 
}

def convert_annotations(dataset: str, type: str):

    src_dir = os.path.join('datasets', dataset, type, "annotations")
    img_dir = os.path.join('datasets', dataset, type, "images")
    dist_dir = os.path.join('datasets', dataset, type, "labels")

    os.makedirs(dist_dir, exist_ok=True)

    print(f"Source annotations: {src_dir}")
    print(f"Source images:      {img_dir}")
    print(f"Output labels:      {dist_dir}")

    for file in os.listdir(src_dir):
        if not file.endswith(".txt"):
            continue
        
        annotation_path = os.path.join(src_dir, file)
        image_path = os.path.join(img_dir, file.replace(".txt", ".jpg"))

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: image not found for {image_path}, skipping.")
            continue

        height_img, width_img = image.shape[:2]

        yolo_annotation = []

        with open(annotation_path, "r") as f:
            lineno = 0
            for line in f:
                try:
                    cleaned = line.strip().rstrip(',')   # removes trailing comma(s)
                    parts = cleaned.split(',')
                    if len(parts) != 8:
                        raise ValueError(f"Expected 8 fields, got {len(parts)}")

                    x, y, w, h, score, category, truncated, occluded = map(int, cleaned.split(','))

                    if category not in CLASS_MAP:
                        continue

                    cls = CLASS_MAP[category]

                    # Convert to YOLO format
                    x_center = (x + w / 2) / width_img
                    y_center = (y + h / 2) / height_img
                    w_norm = w / width_img
                    h_norm = h / height_img

                    lineno += 1
                    yolo_annotation.append(f"{cls} {x_center:.8f} {y_center:.8f} {w_norm:.8f} {h_norm:.8f}")
                except Exception as e:
                    print("\n❌ ERROR WHILE PROCESSING ANNOTATION\n")
                    print(f"File:     {annotation_path}")
                    print(f"Line No.: {lineno}")
                    print(f"Line:     {repr(line)}")
                    print(f"Error:    {e}")
                    raise

        with open(os.path.join(dist_dir, file), "w") as f:
            f.write("\n".join(yolo_annotation))

if __name__== "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument(
             "--type",
             type=str,
             required=True,
             choices=["train", "test", "valid"],
             help="Dataset split (train/test/valid)"
        )

        parser.add_argument(
             "--dataset",
             type=str,
             required=True,
             help="Name of the dataset to be used, e.g. VisDrone-DET"
        )

        args = parser.parse_args()

        convert_annotations(args.dataset, args.type)