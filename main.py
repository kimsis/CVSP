import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from static_object_filter import StaticObjectFilter

def list_cameras(max_cameras=10):
    """List all available cameras with details"""
    available = []
    print("\n" + "="*80)
    print("AVAILABLE CAMERAS:")
    print("="*80)
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                backend = cap.getBackendName()
                
                print(f"\n[{len(available) + 1}] Camera Index: {i}")
                print(f"    Resolution: {width}x{height}")
                print(f"    FPS: {fps}")
                print(f"    Backend: {backend}")
                
                available.append(i)
            cap.release()
    
    print("\n" + "="*80)
    
    if not available:
        print("No cameras found!")
        return None
    
    # Let user select
    while True:
        try:
            choice = input(f"\nSelect camera [1-{len(available)}] or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            choice = int(choice)
            if 1 <= choice <= len(available):
                selected_index = available[choice - 1]
                print(f"\nSelected: Camera Index {selected_index}")
                return selected_index
            else:
                print(f"Please enter a number between 1 and {len(available)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

def draw_label(frame, label, position, color, font_scale=0.6, thickness=2, 
               bg_color=None, text_color=(255, 255, 255)):
    """
    Draw text label with background rectangle.
    
    Args:
        frame: Image to draw on
        label: Text to display
        position: Tuple (x, y) - bottom-left corner of where label should appear
        color: Color for background rectangle (B, G, R)
        font_scale: Size of the font (default 0.6)
        thickness: Thickness of text (default 2)
        bg_color: Background color (if None, uses 'color')
        text_color: Text color (default white)
    """
    x, y = position
    
    # Calculate text size
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Use provided bg_color or default to main color
    background = bg_color if bg_color is not None else color
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x, y - label_height - baseline - 5),
        (x + label_width, y),
        background,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        label,
        (x, y - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness
    )

def get_video_sources(args):
    """
    Get list of video sources based on arguments.
    
    Returns:
        list: List of video sources (file paths or camera index)
    """
    if args.video_folder:
        folder = Path(args.video_folder)
        video_files = sorted(folder.glob("*.mp4"))
        if not video_files:
            print(f"No MP4 files found in {args.video_folder}")
            return []
        print(f"Found {len(video_files)} video(s)")
        return [str(v) for v in video_files]
    elif args.video_path:
        # Single video or camera
        if args.video_path == '0':
            camera_index = list_cameras()
            return [camera_index] if camera_index is not None else []
        return [args.video_path]
    else:
        return []

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
        video_name = Path(str(video_source)).name if not isinstance(video_source, int) else f"Camera {video_source}"
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
            output_name = f'output_{video_name}' if not isinstance(video_source, int) else 'output_camera.mp4'
            out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

        print(f"Starting tracking for {video_name}... Press 'n' to skip to next video, 'q' to quit.")

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

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

                        if is_moving:
                            active_ids.append(track_id)
                            x1, y1, x2, y2 = map(int, box)

                            color = tuple(map(int, colors[track_id % len(colors)]))

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            label = f'ID: {track_id} ({conf:.2f})'

                            draw_label(frame, label, (x1, y1), color)

                    static_filter.cleanup_old_tracks(active_ids)

            cv2.imshow(f'Person Tracking - {video_name}', frame)
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