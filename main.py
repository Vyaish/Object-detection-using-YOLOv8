from ultralytics import YOLO
import cv2
import argparse
import os

def run_image(model, image_path):
    print(f"[INFO] Processing image: {image_path}")

    results = model(image_path)

    # Show result
    results[0].show()

    # Save output
    output_path = "output.jpg"
    results[0].save(filename=output_path)

    print(f"[INFO] Output saved to {output_path}")

    # Count people (class 0 = person in COCO)
    person_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)
    print(f"[INFO] People detected: {person_count}")


def run_video(model, video_path):
    print(f"[INFO] Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_video.mp4", fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        # Count people
        person_count = sum(1 for c in results[0].boxes.cls if int(c) == 0)

        # Add text on frame
        cv2.putText(annotated_frame,
                    f"People Count: {person_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Show
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Save video
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("[INFO] Output video saved as output_video.mp4")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["image", "video"],
                        help="Run mode: image or video")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to image or video file")

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("[ERROR] File not found!")
        return

    print("[INFO] Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # lightweight model

    if args.mode == "image":
        run_image(model, args.path)
    elif args.mode == "video":
        run_video(model, args.path)


if __name__ == "__main__":
    main()

