import cv2
from ultralytics import YOLO

# 모델 로드
model = YOLO("best_chick.pt")

# 비디오 파일 열기
cap = cv2.VideoCapture(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230106081037.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 각 프레임에 대해 예측 수행
    results = model.predict(frame, imgsz=1280, conf=0.3, iou=0.5)

    # 결과 시각화 (프레임에 박스 그리기)
    annotated_frame = results[0].plot()

    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
