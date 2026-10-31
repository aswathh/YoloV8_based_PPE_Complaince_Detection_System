import sys, time
from pathlib import Path
from ultralytics import YOLO

DATA = "Dataset/data.yaml"
AUG = dict(fliplr=0.5, degrees=10, hsv_v=0.4, hsv_s=0.5, scale=0.5)

def train(weights="yolov8s.pt", name="yolov8s_50", epochs=50, batch=8):
    t0 = time.time()
    YOLO(weights).train(data=DATA, epochs=epochs, imgsz=640, batch=batch,
                        workers=2, project="runs", name=name, **AUG)
    print(f"Training time: {(time.time() - t0) / 3600:.2f} hours")

def validate(weights, split="test"):
    m = YOLO(weights).val(data=DATA, split=split, plots=False)
    print(f"P={m.box.mp:.3f} R={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")

if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(*sys.argv[2:])
    else:
        validate(*sys.argv[2:])