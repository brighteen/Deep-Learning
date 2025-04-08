import cv2
import numpy as np
import time
import torch

# --- 1. YOLO 모델 로딩 (best_chick.pt) ---
# 이 부분은 사용 중인 프레임워크에 따라 다르게 구현될 수 있습니다.
# 여기서는 PyTorch 기반 모델이라고 가정합니다.
# 모델 구조와 전처리/후처리 코드는 상황에 맞게 수정하세요.
model = torch.load('best_chick.pt', map_location=torch.device('cpu'))
model.eval()  # 평가 모드

def detect_chickens(frame):
    """
    YOLO 모델을 이용해 frame에서 닭 영역(bbox 리스트)을 검출.
    각 bbox는 (xmin, ymin, xmax, ymax) 형식이라고 가정.
    실제 구현 시 전처리 및 후처리 과정을 추가해야 합니다.
    """
    # 모델 전처리 (예: resize, normalization 등)
    # 이 예제에서는 dummy 함수로, 테스트를 위해 고정된 bbox 목록을 반환합니다.
    # 실제 환경에서는 model(frame)과 관련 후처리 코드가 필요합니다.
    # 예: outputs = model(preprocessed_frame) 후, non-max suppression 적용 등.
    height, width = frame.shape[:2]
    # 임의의 예시로 전체 프레임의 중앙 200×200 영역을 닭 영역으로 반환
    xmin = width // 2 - 100
    ymin = height // 2 - 100
    xmax = xmin + 200
    ymax = ymin + 200
    return [(xmin, ymin, xmax, ymax)]

# --- 2. MHI 관련 파라미터 설정 ---
duration = 1.0    # 모션 이력을 유지하는 시간 (초); 논문에서는 동작 지속시간 관련 상수로 사용
timestamp = 0
mhi_threshold = 30  # 모션이 없는 것으로 간주할 임계값 (조절 필요)

# ROI 별 MHI 이미지를 저장하기 위한 딕셔너리
# 키: bbox tuple, 값: {'mhi': mhi_image, 'prev_gray': 이전 ROI gray frame, 'last_update': timestamp}
roi_MHI = {}

# --- 3. 영상 처리 시작 ---
cap = cv2.VideoCapture('input_video.mp4')  # 육계농장 영상 파일 (또는 카메라)

prev_time = time.time()
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    timestamp = time.time() - prev_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3-1. YOLO를 통한 닭 검출 (bounding boxes)
    bboxes = detect_chickens(frame)  # 각 bbox: (xmin, ymin, xmax, ymax)
    
    # 3-2. 각 닭 영역(ROI)에서 모션 분석 및 MHI 업데이트
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        roi = gray[ymin:ymax, xmin:xmax]  # 현재 ROI (gray scale)
        h, w = roi.shape

        # 만약 이 ROI에 대한 MHI가 없다면 초기화
        if bbox not in roi_MHI:
            # MHI는 float32 타입의 배열로 초기화 (0으로 채움)
            mhi = np.zeros((h, w), dtype=np.float32)
            roi_MHI[bbox] = {'mhi': mhi,
                             'prev_gray': roi.copy(),
                             'last_update': timestamp}
        else:
            mhi = roi_MHI[bbox]['mhi']
            prev_roi = roi_MHI[bbox]['prev_gray']
            
            # 3-2-1. ROI 내에서 차영상(절대 차이) 계산
            diff = cv2.absdiff(roi, prev_roi)
            # 임계값을 적용해 바이너리 모션 맵 생성
            ret_val, motion_mask = cv2.threshold(diff, 30, 1, cv2.THRESH_BINARY)
            
            # 3-2-2. MHI 업데이트: 움직임이 있는 픽셀은 현재 timestamp로 설정,
            # 움직이지 않은 픽셀은 이전 값에서 시간에 따라 감소시킴.
            # OpenCV의 updateMotionHistory 함수를 사용할 수도 있지만, 여기서는 수동 구현.
            # mhi: 각 픽셀에 마지막으로 동작이 검출된 시간을 저장하는 방식.
            # 현재 timestamp를 이용해 업데이트합니다.
            mhi[motion_mask == 1] = timestamp
            # 움직임이 없는 픽셀은 기존 mhi 값 유지하고, duration 초 이전 값은 0으로 만듦.
            mhi[ (timestamp - mhi) > duration ] = 0

            # ROI의 업데이트
            roi_MHI[bbox]['prev_gray'] = roi.copy()
            roi_MHI[bbox]['mhi'] = mhi
            roi_MHI[bbox]['last_update'] = timestamp

        # 3-3. 정적 영역 판단:
        # mhi 값이 0이면 움직임이 없음을 의미하므로, ROI의 해당 영역을 '정적'으로 간주.
        # 단, 전체 ROI에서 움직임이 거의 없으면 (예: 평균 mhi 값 낮으면)
        static_level = cv2.normalize( (duration - mhi), 0, 255, cv2.NORM_MINMAX)
        static_level = np.uint8(static_level)
        # static_level 값이 높을수록 오랫동안 움직임이 없었음을 의미함
        # 예를 들어, 평균 static_level이 200 이상이면 정적 영역으로 판단
        mean_static = np.mean(static_level)

        # 3-4. 결과 표시: ROI에 상태에 따라 테두리 색 또는 텍스트 표시
        color = (0, 255, 0)  # 기본: 움직임 있는 닭 (녹색)
        label = "Active"
        if mean_static > 200:  # 임계값 (조정 필요): 일정 시간 동안 움직임이 없으면
            color = (0, 0, 255)  # 정적이면 빨간색으로 표시 (폐사체 후보)
            label = "Static"

        # 검출된 닭 영역에 상태 및 static_level 영상(옵션) 표시
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ROI 내부의 모션 정보 시각화를 원하는 경우 (예: static_level을 컬러맵 적용)
        static_vis = cv2.applyColorMap(static_level, cv2.COLORMAP_JET)
        cv2.imshow(f"ROI_{bbox}", static_vis)

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키 종료
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
