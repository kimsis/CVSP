"""
Filter ultralytics COCO dataset to person-only.
Handles multiple splits (train/val/test) and updates labels, images, and annotations.
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm


def filter_coco_split(
    images_dir,
    labels_dir,
    index_file=None,
    backup_dir=None,
    annotations_json=None,
    person_class_id=0,
    person_category_id=1,
    verbose=True
):
    """
    Filter YOLO format dataset split to only person class.
    Also filters corresponding COCO JSON annotations if provided.
    
    Args:
        images_dir: Path to images directory (e.g., datasets/coco/images/train)
        labels_dir: Path to labels directory (e.g., datasets/coco/labels/train)
        index_file: Optional index file to update
        backup_dir: Optional backup directory for removed images
        annotations_json: Optional path to instances_*.json to filter
        person_class_id: YOLO person class ID (default 0)
        person_category_id: COCO person category ID (default 1)
        verbose: Print progress information
    """
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else None
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if labels_dir and not labels_dir.exists():
        labels_dir = None  # No labels for this split
    
    if index_file:
        index_file = Path(index_file)
    
    split_name = images_dir.name
    has_labels = labels_dir is not None
    
    print(f"\n{'='*70}")
    print(f"FILTERING {split_name.upper()} SPLIT")
    print(f"{'='*70}")
    
    # STEP 1: Scan files
    print(f"\n[1/4] Scanning {split_name} dataset...")
    all_images = list(images_dir.glob('*'))
    image_files = [f for f in all_images if f.is_file()]
    all_labels = list(labels_dir.glob('*.txt')) if has_labels else []
    
    original_img_count = len(image_files)
    original_label_count = len(all_labels)
    
    if verbose:
        print(f"  ├─ Images: {original_img_count:,}")
        if has_labels:
            print(f"  ├─ Labels: {original_label_count:,}")
        else:
            print(f"  ├─ Labels: None (no labels in this split)")
        if index_file and index_file.exists():
            index_lines = len(index_file.read_text().strip().split('\n'))
            print(f"  └─ Index entries: {index_lines:,}")
    
    # STEP 2: Filter labels (if they exist)
    images_to_keep = set()
    labels_to_remove = []
    kept_labels = 0
    
    if has_labels:
        print(f"\n[2/4] Filtering labels to person class (ID={person_class_id})...")
        
        for label_file in tqdm(all_labels, desc="  Processing labels"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            has_person = False
            person_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id == person_class_id:
                        has_person = True
                        person_lines.append(line)
            
            if has_person:
                with open(label_file, 'w') as f:
                    f.writelines(person_lines)
                
                img_name = label_file.stem
                images_to_keep.add(img_name)
            else:
                labels_to_remove.append(label_file)
        
        kept_labels = len(all_labels) - len(labels_to_remove)
        
        if verbose:
            print(f"  ├─ Removed: {len(labels_to_remove):,} labels")
            print(f"  └─ Kept: {kept_labels:,} labels")
        
        # STEP 3: Remove non-person labels
        print(f"\n[3/4] Removing non-person label files...")
        for label_file in tqdm(labels_to_remove, desc="  Deleting"):
            label_file.unlink()
    else:
        # No labels, so keep all images initially (will be filtered by annotations if provided)
        print(f"\n[2/4] No labels to filter (skipping)...")
        images_to_keep = {img.stem for img in image_files}
    
    # STEP 4: Remove non-person images
    print(f"\n[4/4] Removing non-person images...")
    
    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    removed_img_count = 0
    for img_file in tqdm(image_files, desc="  Processing"):
        img_stem = img_file.stem
        
        if img_stem not in images_to_keep:
            if backup_dir:
                shutil.move(str(img_file), str(backup_dir / img_file.name))
            else:
                img_file.unlink()
            removed_img_count += 1
    
    remaining_img_count = len(list(images_dir.glob('*')))
    
    # STEP 5: Update index file
    if index_file and index_file.exists():
        print(f"\n[5/5] Updating index file...")
        
        with open(index_file, 'r') as f:
            index_lines = f.readlines()
        
        filtered_lines = []
        for line in index_lines:
            img_path = line.strip()
            if img_path:
                img_stem = Path(img_path).stem
                if img_stem in images_to_keep:
                    filtered_lines.append(line)
        
        with open(index_file, 'w') as f:
            f.writelines(filtered_lines)
        
        if verbose:
            print(f"  ├─ Original: {len(index_lines):,} entries")
            print(f"  └─ Kept: {len(filtered_lines):,} entries")
    
    # STEP 5: Filter annotations JSON if provided
    if annotations_json:
        filter_coco_annotations_json(
            annotations_json=annotations_json,
            images_to_keep=images_to_keep,
            person_category_id=person_category_id,
            verbose=verbose
        )
    
    # Summary
    print(f"\n[Summary] {split_name.upper()}")
    print(f"  Images:  {original_img_count:,} → {remaining_img_count:,} (removed {removed_img_count:,})")
    if has_labels:
        print(f"  Labels:  {original_label_count:,} → {kept_labels:,} (removed {len(labels_to_remove):,})")
    if annotations_json:
        print(f"  Annotations JSON: Updated")
    
    return {
        'split': split_name,
        'images_kept': remaining_img_count,
        'labels_kept': kept_labels,
        'images_removed': removed_img_count,
        'labels_removed': len(labels_to_remove)
    }


def filter_coco_annotations_json(
    annotations_json,
    images_to_keep,
    output_json=None,
    person_category_id=1,
    verbose=True
):
    """
    Filter COCO annotations JSON to keep only person class and specified images.
    
    Args:
        annotations_json: Path to instances_*.json file
        images_to_keep: Set/list of image filenames or IDs to keep
        output_json: Where to save filtered JSON (default: overwrite)
        person_category_id: COCO person category ID (default 1)
        verbose: Print progress
    """
    
    annotations_json = Path(annotations_json)
    if not annotations_json.exists():
        print(f"⚠️  Annotations file not found: {annotations_json}")
        return
    
    print(f"\n{'='*70}")
    print("FILTERING ANNOTATIONS JSON")
    print(f"{'='*70}")
    
    print(f"\n[1/3] Loading {annotations_json.name}...")
    with open(annotations_json, 'r') as f:
        coco_data = json.load(f)
    
    original_imgs = len(coco_data['images'])
    original_anns = len(coco_data['annotations'])
    
    print(f"  ├─ Images: {original_imgs:,}")
    print(f"  ├─ Annotations: {original_anns:,}")
    print(f"  └─ Categories: {len(coco_data['categories'])}")
    
    # Filter annotations to person only
    print(f"\n[2/3] Filtering to person class (ID={person_category_id})...")
    person_anns = [
        ann for ann in coco_data['annotations']
        if ann['category_id'] == person_category_id
    ]
    print(f"  └─ Kept: {len(person_anns):,} annotations")
    
    # Get image IDs from filtered annotations
    image_ids_with_person = {ann['image_id'] for ann in person_anns}
    
    # If images_to_keep is provided (set of image stems), also filter by those
    if images_to_keep:
        # Map image stems to IDs in the JSON
        ids_from_stems = set()
        for img in coco_data['images']:
            img_stem = Path(img['file_name']).stem
            if img_stem in images_to_keep:
                ids_from_stems.add(img['id'])
        # Intersect: keep only images that have person annotations AND are in images_to_keep
        image_ids_with_person = image_ids_with_person.intersection(ids_from_stems)
    
    # Keep only those images
    print(f"\n[3/3] Filtering to images with person annotations...")
    coco_data['images'] = [
        img for img in coco_data['images']
        if img['id'] in image_ids_with_person
    ]
    coco_data['annotations'] = person_anns
    coco_data['categories'] = [
        cat for cat in coco_data['categories']
        if cat['id'] == person_category_id
    ]
    
    kept_imgs = len(coco_data['images'])
    print(f"  └─ Kept: {kept_imgs:,} images")
    
    # Save
    output_json = output_json or annotations_json
    output_json = Path(output_json)
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"\n[Summary] Annotations")
    print(f"  Images:      {original_imgs:,} → {kept_imgs:,}")
    print(f"  Annotations: {original_anns:,} → {len(person_anns):,}")
    print(f"  Saved to: {output_json.name}")


if __name__ == "__main__":
    # Example usage for ultralytics COCO structure
    # Only val annotations exist
    
    coco_root = Path('datasets/coco')
    
    # Filter each split
    results = {}
    
    for split in ['train', 'val', 'test']:
        # Check if annotations JSON exists for this split
        annotations_json = coco_root / 'annotations' / f'instances_{split}.json'
        if split == 'val' and annotations_json.exists():
            # Val split has annotations, filter them
            annotations_path = annotations_json
        else:
            annotations_path = None
        
        result = filter_coco_split(
            images_dir=coco_root / 'images' / split,
            labels_dir=coco_root / 'labels' / split,
            index_file=coco_root / f'{split}.txt' if (coco_root / f'{split}.txt').exists() else None,
            backup_dir=coco_root / 'backup' / split,
            annotations_json=annotations_path,
            person_class_id=0,
            person_category_id=1,
            verbose=True
        )
        results[split] = result
    
    # Final summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    for split, res in results.items():
        print(f"{split.upper():6} - Images: {res['images_kept']:,} | Labels: {res['labels_kept']:,}")
    print(f"{'='*70}")
    print("✓ Filtering complete!")
