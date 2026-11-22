import argparse
import cv2
import torch
from ultralytics import YOLO


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_detection(weights_path: str, source: str, conf: float):
    device = get_device()
    print(f"Running inference on device: {device}")

    model = YOLO(weights_path)
    video_source = int(source) if source.isdigit() else source

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, device=device, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Helmet Detection (macOS)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Helmet Detection for macOS")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Path to model weights (e.g., runs/detect/helmet_train/weights/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="0 for Mac webcam, or path to image/video file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (0.0 - 1.0)",
    )
    args = parser.parse_args()
    run_detection(args.weights, args.source, args.conf)
