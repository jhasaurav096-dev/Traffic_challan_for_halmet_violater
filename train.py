import torch
from ultralytics import YOLO


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = get_device()
    print(f"Starting YOLOv8 training on device: {device}")

    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project="runs/detect",
        name="helmet_train",
    )


if __name__ == "__main__":
    main()
