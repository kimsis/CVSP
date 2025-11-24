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

def convert_annotations(dataset: str, type: str, source_format: str):

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
        if source_format == "image":
            image_path = os.path.join(img_dir, file.replace(".txt", ".jpg"))
        else:
            image_path = os.path.join(img_dir, file.replace(".txt", "/0000001.jpg"))

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: image not found for {image_path}, skipping.")
            continue

        height_img, width_img = image.shape[:2]

        base_name = os.path.splitext(file)[0]
        label_dir = os.path.join(dist_dir, base_name)
        os.makedirs(label_dir, exist_ok=True)

        if source_format == "image":
            process_file_image(annotation_path, dist_dir, file, width_img, height_img)
        else:
            process_file_video(annotation_path, label_dir, base_name, width_img, height_img)

def parse_line(line, expected_fields):
    """
    Parse a comma-separated VisDrone line,
    strip trailing commas, validate length, return list of ints.
    """
    cleaned = line.strip().rstrip(',')
    parts = cleaned.split(',')

    if len(parts) != expected_fields:
        raise ValueError(f"Expected {expected_fields} fields, got {len(parts)}")

    return list(map(int, parts))

def convert_box(x, y, w, h, width_img, height_img):
    """
    Convert VisDrone box to YOLO normalized format.
    """
    x_center = (x + w / 2) / width_img
    y_center = (y + h / 2) / height_img
    w_norm   = w / width_img
    h_norm   = h / height_img
    return x_center, y_center, w_norm, h_norm

def process_file_image(annotation_path, dist_dir, out_filename, width_img, height_img):
    """Process VisDrone image-level annotation files (8 fields)."""

    yolo = []

    with open(annotation_path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            try:
                x, y, w, h, score, category, truncated, occluded = parse_line(line, expected_fields=8)

                if category not in CLASS_MAP:
                    continue

                cls = CLASS_MAP[category]
                x_center, y_center, w_norm, h_norm = convert_box(x, y, w, h, width_img, height_img)

                yolo.append(f"{cls} {x_center:.8f} {y_center:.8f} {w_norm:.8f} {h_norm:.8f}")

            except Exception as e:
                print("\n❌ ERROR WHILE PROCESSING ANNOTATION")
                print(f"File:     {annotation_path}")
                print(f"Line No.: {lineno}")
                print(f"Line:     {repr(line)}")
                print(f"Error:    {e}")
                raise

    # Write final file
    out_path = os.path.join(dist_dir, out_filename)
    with open(out_path, "w") as f:
        f.write("\n".join(yolo))

def process_file_video(annotation_path, dist_dir, base_name, width_img, height_img):
    """Process VisDrone video annotation files (10 fields)."""

    frame_annotations = {}

    with open(annotation_path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            try:
                frame, target, x, y, w, h, score, category, truncated, occluded = \
                    parse_line(line, expected_fields=10)

                if category not in CLASS_MAP:
                    continue

                cls = CLASS_MAP[category]
                x_center, y_center, w_norm, h_norm = convert_box(x, y, w, h, width_img, height_img)

                yolo_line = f"{cls} {x_center:.8f} {y_center:.8f} {w_norm:.8f} {h_norm:.8f}"
                frame_annotations.setdefault(frame, []).append(yolo_line)

            except Exception as e:
                print("\n❌ ERROR WHILE PROCESSING ANNOTATION")
                print(f"File:     {annotation_path}")
                print(f"Line No.: {lineno}")
                print(f"Line:     {repr(line)}")
                print(f"Error:    {e}")
                raise

    # Write one label file per frame
    for frame_num, labels in frame_annotations.items():
        out_name = f"{frame_num:07d}.txt"
        out_path = os.path.join(dist_dir, out_name)
        with open(out_path, "w") as f:
            f.write("\n".join(labels))

if __name__== "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument(
             "--type",
             type=str,
             required=True,
             choices=["train", "test", "val"],
             help="Dataset split (train/test/val)"
        )

        parser.add_argument(
             "--dataset",
             type=str,
             required=True,
             help="Name of the dataset to be used, e.g. VisDrone-DET"
        )

        parser.add_argument(
            "--source-format",
            type=str,
            required=False,
            choices=["image", "video"],
            default="image",
            help="Format of the source data (image/video)",
        )

        args = parser.parse_args()

        convert_annotations(args.dataset, args.type, args.source_format)