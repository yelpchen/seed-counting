from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11n-seg.pt")
    model.train(
        data="seedcount.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        patience=15,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )