from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt") # 원하는 크기 모델 입력(n ~ x)

# result = model.predict("./test4.jpg", save=False, conf=0.5) # conf=0.5 : 신뢰도가 0.5 이상인 것만 출력
result = model.predict("test4.jpg")
plots = result[0].plot()

# 박스 정보 확인(좌표, 신뢰도, 클래스)
boxes = result[0].boxes

for box in boxes :
    print(box.xyxy.cpu().detach().numpy().tolist()) # 출력값을 tebsor에서 numpy로 변환
    print(box.conf.cpu().detach().numpy().tolist())
    print(box.cls.cpu().detach().numpy().tolist())

cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()