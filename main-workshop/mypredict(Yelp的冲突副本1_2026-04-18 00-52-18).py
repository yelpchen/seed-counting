from ultralytics import YOLO

model = YOLO(r"G:\YOLO\ultralytics-8.3.163\runs\segment\train9\weights\best.pt")
model.predict(
    source=r"C:\Users\10761\Desktop\毕业设计\3.png",
    # source=0,
    save=True,
    show=False,
)