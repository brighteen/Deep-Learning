import cv2
import numpy as np
from ultralytics import YOLO

import torch

# 1. YOLO 모델 로드
model = model = YOLO(r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt')

# 워크어라운드: Detect 모듈에 grid 속성이 없으면 임의의 grid를 추가
for module in model.model.modules():
    if module.__class__.__name__ == 'Detect':
        if not hasattr(module, 'grid'):
            # module.stride 리스트 길이에 맞춰 grid를 초기화합니다.
            module.grid = [torch.zeros(1)] * len(module.stride)

# 2. 입력 영상 열기
input_video_path = r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\MotionHistoryImage(MHI)\tile_r0_c3.mp4'
cap = cv2.VideoCapture(input_video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video_path = 'output_video.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 프레임 추론 수행
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # 4. 이진 마스크 생성: 닭 객체만 흰색, 나머지는 검정
    mask = np.zeros((height, width), dtype=np.uint8)
    for det in detections:
        xmin, ymin, xmax, ymax, conf, cls = det
        # 좌표 정수형 변환
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color=255, thickness=-1)

    out.write(mask)
    cv2.imshow('Binarized Chick Mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
