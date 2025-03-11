import torch
import cv2

# Torch Hub를 통해 YOLOv5s 모델 로드
model_v5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 추론 수행
results = model_v5('test.jpeg')
results.print()  # 결과 요약 출력

# 결과 시각화 (결과 이미지를 저장하거나 렌더링)
img_v5 = results.render()[0]
cv2.imshow("YOLOv5s", img_v5)
cv2.waitKey(0)
cv2.destroyAllWindows()
