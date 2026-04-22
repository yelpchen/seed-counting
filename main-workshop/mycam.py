from ultralytics import YOLO

import cv2

model = YOLO(r'yolo11n.pt')
results = model(
    source=0,
    stream=True,
)

for result in results:
    plotted = result.plot()
    cv2.imshow('result', plotted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break