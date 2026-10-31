from pathlib import Path
import cv2
from ultralytics import YOLO
from src.compliance import check_workers, dashboard

MODEL_PATH = Path("models/best.pt")

def load_model(path=MODEL_PATH):
    return YOLO(str(path))

def run_image(model, image_path, conf=0.25, save_dir=None):
    r = model.predict(image_path, conf=conf, verbose=False)[0]
    statuses = check_workers(r)
    annotated = r.plot()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(save_dir) / Path(image_path).name), annotated)
    return annotated, statuses

if __name__ == "__main__":
    import sys
    model = load_model()
    _, statuses = run_image(model, sys.argv[1], save_dir="outputs/predictions")
    print(statuses, dashboard(statuses))