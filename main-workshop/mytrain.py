from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo11n-seg.pt")
    model.train(
        data = "myseed.yaml",
        epochs = 100,
        imgsz=640,
        batch=16,
        cache=False,
        workers=1,
    )