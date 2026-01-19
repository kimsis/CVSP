import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from static_object_filter import StaticObjectFilter
from video_utils import draw_label, resize_for_display, get_video_sources

def main(args: argparse.Namespace):
    model_path = Path(f"runs/detect/{args.trained_model}/weights/best.pt")
    model = YOLO(model_path)

    # Get video sources
    video_sources = get_video_sources(args)
    if not video_sources:
        print("No video source specified or found")
        return

    # Initialize filter and colors ONCE
    static_filter = StaticObjectFilter(
        frames_threshold=30,
        movement_threshold=10
    )
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    # Process each video
    for i, video_source in enumerate(video_sources, 1):
        is_camera = isinstance(video_source, int)
        video_name = Path(str(video_source)).name if not is_camera else f"Camera {video_source}"
        print(f"\n[{i}/{len(video_sources)}] Processing: {video_name}")

        capture = cv2.VideoCapture(video_source)

        if not capture.isOpened():
            print(f"Error: Could not open {video_source}")
            continue

        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate delay for proper FPS playback (milliseconds)
        # Use 30 FPS if video FPS is 0 or invalid
        target_fps = 30
        delay = int(1000 / target_fps) if target_fps > 0 else 1

        save_output = args.save_output
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_name = f'output_{video_name}' if not is_camera else 'output_camera.mp4'
            out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

        print(f"Starting tracking for {video_name}... Press 'n' to skip to next video, 'q' to quit.")

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640)) # Resize for faster processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

            results = model.track(frame, True, True, verbose=False, max_det=300, conf=0.3, iou=0.7)

            for result in results:
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    confidence = result.boxes.conf.cpu().numpy()

                    # Draw bboxes and labels
                    active_ids = []
                    for box, track_id, conf in zip(boxes, track_ids, confidence):
                        is_moving = static_filter.update(track_id, box)

                        # Only track and render if object is moving
                        if is_moving:
                            active_ids.append(track_id)
                            x1, y1, x2, y2 = map(int, box)

                            color = tuple(map(int, colors[track_id % len(colors)]))

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            label = f'ID: {track_id} ({conf:.2f})'

                            draw_label(frame, label, (x1, y1), color)

                    static_filter.cleanup_old_tracks(active_ids)

            # Resize frame for display if needed
            display_frame = resize_for_display(frame, max_width=1920, max_height=1080)
            
            cv2.imshow(f'Person Tracking - {video_name}', display_frame)
            if save_output:
                out.write(frame)
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('n'):  # Skip to next video
                break
            elif key == ord('q'):  # q - quit completely
                capture.release()
                if save_output:
                    out.release()
                cv2.destroyAllWindows()
                print("\nTracking ended by user!")
                return

        capture.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()  # Close window for this video
        
        # Small delay to ensure window closes properly
        cv2.waitKey(1)

    print("\nAll videos processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--video_path", type=str, required=False, default="0", help="Path to video or flag for camera")
    parser.add_argument("--video_folder", type=str, required=False, help="Folder containing MP4 videos")
    parser.add_argument("--save_output", type=bool, required=False, default=False, help="Flag to save output video")

    args = parser.parse_args()
    main(args)