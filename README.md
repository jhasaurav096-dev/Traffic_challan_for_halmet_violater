# YOLOv8 Helmet Detection (macOS & VS Code)

Complete project template configured for macOS (Apple Silicon `mps` & Intel CPU) and Visual Studio Code.

## 1. Setup Environment in VS Code
Open this folder in VS Code, open the terminal (`Ctrl + ~`), and run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset Placement
Place your YOLO-formatted dataset inside `dataset/`:
- `dataset/train/images/` & `dataset/train/labels/`
- `dataset/val/images/` & `dataset/val/labels/`

## 3. Train Model
```bash
python train.py
```

## 4. Run Real-Time Detection
Run with Mac webcam (`source=0`):
```bash
python detect.py --weights runs/detect/helmet_train/weights/best.pt --source 0
```
Or press `F5` in VS Code to launch via the preconfigured debugger. Press `q` to quit the preview window.
