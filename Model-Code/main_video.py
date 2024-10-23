import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import numpy as np

video_path = '/content/drive/MyDrive/YoloV8New/1.mp4'

model_plate_detection = YOLO("/content/drive/MyDrive/YoloV8New/runs/detect/plate_train/weights/best.pt")
model_plate_ocr = YOLO("/content/drive/MyDrive/YoloV8New/ocr_dataset/runs/detect/plate_train_denormal/weights/best.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video açılamadı!")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        plate_results = model_plate_detection(frame)

        for plate_result in plate_results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate_result
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            plate_image = frame[y1:y2, x1:x2]

            ocr_results = model_plate_ocr(plate_image)

            plate_data = []

            for ocr_result in ocr_results[0].boxes.data.tolist():
                ocr_x1, ocr_y1, ocr_x2, ocr_y2, ocr_score, ocr_class_id = ocr_result
                ocr_x1, ocr_y1, ocr_x2, ocr_y2, ocr_class_id = int(ocr_x1), int(ocr_y1), int(ocr_x2), int(ocr_y2), int(ocr_class_id)

                center_x = (ocr_x1 + ocr_x2) // 2
                center_y = (ocr_y1 + ocr_y2) // 2
                plate_data.append((center_x, center_y, ocr_class_id))

            def sort_by_x(data):
                return sorted(data, key=lambda x: x[0])

            sorted_plate_data = sort_by_x(plate_data)
            plate_text = "".join([ocr_results[0].names[char_data[2]] for char_data in sorted_plate_data])
            print("Plate: ", end="")
            print(plate_text)

        cv2_imshow(frame)

cap.release()
cv2.destroyAllWindows()
