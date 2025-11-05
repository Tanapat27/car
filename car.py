from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")  


image_path = "car.png"  
image = cv2.imread(image_path)


results = model(image)


detections = results[0].boxes.data  
names = model.names  

car_count = 0


for box in detections:
    x1, y1, x2, y2, conf, cls = box
    label = names[int(cls)]
    if label in ["car", "truck", "bus", "motorbike"]:  
        car_count += 1
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


print(f"จำนวนรถยนต์ที่ตรวจพบ: {car_count} คัน")


cv2.imshow("ผลการตรวจจับ", image)
cv2.waitKey(0)
cv2.destroyAllWindows()