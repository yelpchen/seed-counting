from ultralytics import YOLO

model = YOLO(r"G:\YOLO\ultralytics-8.3.163\runs\segment\train12\weights\best.pt")
model.predict(
    source=r"G:\YOLO\毕业设计\seeds",
    # source=0,
    save=True,
    show=False,
)