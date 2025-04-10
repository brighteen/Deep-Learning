import cv2
import numpy as np
from ultralytics import YOLO
import time

# 1. YOLO 모델 로드 (Ultralytics YOLO, best_chick.pt)
model = YOLO(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt")

# 2. 영상 소스 열기
video_path = r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\tile_r0_c4.mp4'  # 영상 파일 경로 또는 카메라 번호 (예: 0)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    exit()

# 3. 첫 프레임 읽기 및 기본 설정
ret, prev_frame = cap.read()
if not ret:
    print("첫 번째 프레임을 읽지 못했습니다.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
h, w = prev_gray.shape
mhi = np.zeros((h, w), dtype=np.float32)  # MHI 이미지 초기화

# MHI 업데이트 파라미터
duration = 1.0    # MHI 유지 시간 (초)
timestamp = 0.0
dt = 1/30.0       # 약 30 FPS 기준

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()  # 타임스탬프 계산을 위한 시작 시간
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    timestamp += dt

    # 4. YOLO 모델로 닭 객체 검출 (Ultralytics API 사용)
    # 모델은 BGR 이미지도 내부적으로 처리하는 경우가 있으므로 바로 전달하거나,
    # 필요시 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)로 변환할 수 있습니다.
    results = model(frame)
    
    # YOLO v8의 결과는 리스트 형태로 나오며, 각 요소는 detect 결과 객체입니다.
    # 결과 객체에서 boxes (바운딩 박스 정보)를 가져옵니다.
    yolo_mask = np.zeros_like(gray, dtype=np.uint8)
    
    for result in results:
        boxes = result.boxes  # Boxes 객체 (xyxy, confidence, class 등 정보 포함)
        if boxes is None:
            continue
        for box in boxes:
            # box.xyxy: tensor([[x1, y1, x2, y2]])
            # box.conf: confidence, box.cls: class index (필요 시)
            coords = box.xyxy.cpu().numpy().squeeze()  # 배열로 변환
            if coords.ndim == 0 or len(coords) < 4:
                continue
            x1, y1, x2, y2 = map(int, coords[:4])
            # yolo_mask의 해당 영역에 1 할당 (검출된 닭 영역)
            cv2.rectangle(yolo_mask, (x1, y1), (x2, y2), 1, thickness=-1)

    # 5. 전체 프레임에서 프레임 차분을 이용한 움직임 마스크 생성
    frame_diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask_full = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)

    # 6. YOLO 마스크와 결합하여 닭 영역 내에서 발생한 움직임만 선택
    motion_mask = cv2.bitwise_and(motion_mask_full, yolo_mask)

    # 7. MHI 업데이트 (OpenCV 내장 함수 이용)
    cv2.motempl.updateMotionHistory(motion_mask, mhi, timestamp, duration)

    # 8. MHI 시각화를 위한 정규화 및 MEI 생성
    mhi_normalized = np.uint8(np.clip((mhi - (timestamp - duration)) / duration, 0, 1) * 255)
    _, mei = cv2.threshold(mhi_normalized, 30, 255, cv2.THRESH_BINARY)

    # 9. 결과 영상 출력
    cv2.imshow("Original Frame", frame)
    cv2.imshow("YOLO Mask", yolo_mask * 255)       # 닭 객체 영역 (디버깅 용)
    cv2.imshow("Motion Mask", motion_mask * 255)     # 닭 영역 내 움직임 마스크
    cv2.imshow("MHI", mhi_normalized)                # 누적된 모션 히스토리 이미지
    cv2.imshow("MEI", mei)                           # 이진화된 모션 에너지 이미지

    # 종료 키 처리 ('q' 누르면 종료)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 다음 프레임 처리를 위한 업데이트
    prev_gray = gray.copy()
    
    # 간단한 FPS 조절 (필요 시)
    elapsed_time = time.time() - start_time
    if elapsed_time < dt:
        time.sleep(dt - elapsed_time)

cap.release()
cv2.destroyAllWindows()
