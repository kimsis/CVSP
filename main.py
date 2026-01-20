import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from static_object_filter import StaticObjectFilter
from video_utils import draw_label, resize_for_display, get_video_sources

def run_pipeline(model, video_sources, save_output=False, output_dir=None, 
                 headless=False, model_name=None, collect_metrics=False):
    """
    Run the detection pipeline on videos with a given model.
    
    Args:
        model: YOLO model instance
        video_sources: List of video paths or camera indices
        save_output: Whether to save output videos
        output_dir: Directory to save output videos (if save_output=True)
        headless: Run without display windows
        model_name: Name to display in window title
        collect_metrics: Whether to collect and return performance metrics
    
    Returns:
        dict: Metrics if collect_metrics=True, otherwise None
    """
    if model_name is None:
        model_name = "Detection"

    # Initialize filter and colors ONCE
    static_filter = StaticObjectFilter(
        frames_threshold=30,
        movement_threshold=10
    )
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    # Metrics tracking
    metrics = {
        "model_name": model_name,
        "status": "SUCCESS",
        "videos": [],
        "total_frames": 0,
        "total_detections": 0,
        "total_inference_time": 0.0,
        "avg_fps": 0.0,
        "avg_detections_per_frame": 0.0
    } if collect_metrics else None

    # Process each video
    for i, video_source in enumerate(video_sources, 1):
        is_camera = isinstance(video_source, int)
        video_name = Path(str(video_source)).name if not is_camera else f"Camera_{video_source}"
        print(f"\n[{i}/{len(video_sources)}] Processing: {video_name}")

        capture = cv2.VideoCapture(video_source)

        if not capture.isOpened():
            print(f"Error: Could not open {video_source}")
            continue

        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate delay for proper FPS playback (milliseconds)
        target_fps = 30
        delay = int(1000 / target_fps) if target_fps > 0 else 1

        # Video metrics
        video_metrics = {
            "video_name": video_name,
            "video_source": str(video_source),
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": 0,
            "total_detections": 0,
            "inference_time": 0.0,
            "avg_fps": 0.0
        } if collect_metrics else None
        
        frame_count = 0
        detection_count = 0

        out = None
        if save_output:
            if output_dir:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_path = out_dir / f"{video_name}"
            else:
                output_path = f'output_{video_name}' if not is_camera else 'output_camera.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 640))

        if not headless:
            print(f"Starting tracking for {video_name}... Press 'n' to skip to next video, 'q' to quit.")

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            
            frame_count += 1

            frame = cv2.resize(frame, (640, 640))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

            # Inference with timing
            start_time = time.time()
            results = model.track(frame, True, True, verbose=False, max_det=300, conf=0.3, iou=0.7)
            if collect_metrics:
                video_metrics["inference_time"] += (time.time() - start_time)

            for result in results:
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    confidence = result.boxes.conf.cpu().numpy()
                    
                    if collect_metrics:
                        detection_count += len(boxes)

                    # Draw bboxes and labels (only if showing or saving)
                    if not headless or save_output:
                        active_ids = []
                        for box, track_id, conf in zip(boxes, track_ids, confidence):
                            is_moving = static_filter.update(track_id, box)

                            if is_moving:
                                active_ids.append(track_id)
                                x1, y1, x2, y2 = map(int, box)
                                color = tuple(map(int, colors[track_id % len(colors)]))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                label = f'ID: {track_id} ({conf:.2f})'
                                draw_label(frame, label, (x1, y1), color)

                        static_filter.cleanup_old_tracks(active_ids)

            # Display (unless headless)
            if not headless:
                display_frame = resize_for_display(frame, max_width=1920, max_height=1080)
                cv2.imshow(f'{model_name} - {video_name}', display_frame)
                
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('n'):  # Skip to next video
                    break
                elif key == ord('q'):  # q - quit completely
                    capture.release()
                    if out:
                        out.release()
                    cv2.destroyAllWindows()
                    if collect_metrics:
                        metrics["status"] = "INTERRUPTED"
                        return metrics
                    return
            
            if save_output and out:
                out.write(frame)

        capture.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        # Calculate video metrics
        if collect_metrics:
            video_metrics["total_frames"] = frame_count
            video_metrics["total_detections"] = detection_count
            video_metrics["avg_fps"] = frame_count / video_metrics["inference_time"] if video_metrics["inference_time"] > 0 else 0
            video_metrics["avg_detections_per_frame"] = detection_count / frame_count if frame_count > 0 else 0
            
            metrics["videos"].append(video_metrics)
            metrics["total_frames"] += frame_count
            metrics["total_detections"] += detection_count
            metrics["total_inference_time"] += video_metrics["inference_time"]
            
            print(f"  Frames: {frame_count}, Detections: {detection_count}, Avg FPS: {video_metrics['avg_fps']:.2f}")

    # Calculate overall metrics
    if collect_metrics:
        if metrics["total_frames"] > 0:
            metrics["avg_fps"] = metrics["total_frames"] / metrics["total_inference_time"] if metrics["total_inference_time"] > 0 else 0
            metrics["avg_detections_per_frame"] = metrics["total_detections"] / metrics["total_frames"]
        return metrics
    
    print("\nAll videos processed!")

def _resolve_model_path(model_id: str) -> Path:
    """Return a usable model path; accept full .pt paths or run names."""
    candidate = Path(model_id)
    if candidate.suffix == ".pt" and candidate.exists():
        return candidate
    return Path(f"runs/detect/{model_id}/weights/best.pt")


def main(args: argparse.Namespace):
    """CLI entry that supports one or multiple models."""
    model_ids = []
    if args.trained_models:
        model_ids.extend([m.strip() for m in args.trained_models.split(',') if m.strip()])
    if args.trained_model:
        model_ids.append(args.trained_model)

    if not model_ids:
        print("No trained model provided. Use --trained_model for one or --trained_models for many.")
        return

    video_sources = get_video_sources(args)
    if not video_sources:
        print("No video source specified or found")
        return

    for model_id in model_ids:
        model_path = _resolve_model_path(model_id)
        if not model_path.exists():
            print(f"Skipping {model_id}: {model_path} not found")
            continue

        model = YOLO(model_path)
        run_pipeline(
            model=model,
            video_sources=video_sources,
            save_output=args.save_output,
            output_dir=None,
            headless=False,
            model_name=f"Model: {model_id}",
            collect_metrics=False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model", type=str, required=False, help="Single trained model name or .pt path")
    parser.add_argument("--trained_models", type=str, required=False, help="Comma-separated trained model names")
    parser.add_argument("--video_path", type=str, required=False, default="0", help="Path to video or flag for camera")
    parser.add_argument("--video_folder", type=str, required=False, help="Folder containing MP4 videos")
    parser.add_argument("--save_output", type=bool, required=False, default=False, help="Flag to save output video")

    args = parser.parse_args()
    main(args)