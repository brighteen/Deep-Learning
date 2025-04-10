import cv2
import numpy as np

# 1. 파라미터 설정 및 라이브러리 초기화
video_path = r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\tile_r0_c4.mp4'  # 혹은 카메라의 경우 0
# video_path = r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4'  # 혹은 카메라의 경우 0
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    exit()

# MHI를 위한 파라미터 (duration: MHI가 유지될 시간)
duration = 1.0  # 초 단위, 이후 시간은 디케이(decay)됩니다.
timestamp = 0.0   # 초기 시간
dt = 0.033       # 프레임 당 시간 간격 (약 30 FPS 기준)

# 첫 번째 프레임 읽기 및 그레이스케일 변환
ret, prev_frame = cap.read()
if not ret:
    print("첫 번째 프레임을 읽지 못했습니다.")
    cap.release()
    exit()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 영상 크기에 맞추어 MHI 이미지를 초기화 (float32 타입)
h, w = prev_gray.shape
mhi = np.zeros((h, w), np.float32)

# (선택 사항) ROI 설정: 전체 영상 대신 관심 영역만 처리하고 싶을 경우
# roi = cv2.selectROI("Select ROI", prev_frame, False, False)
# x, y, w_roi, h_roi = roi
# cv2.destroyWindow("Select ROI")

# 2. 프레임 반복 처리
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 타임스탬프 업데이트
    timestamp += dt

    # (ROI 적용 시) frame = frame[y:y+h_roi, x:x+w_roi]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. 프레임 차분을 이용한 움직임 검출
    # 이전 프레임과의 차이 계산 및 임계값 적용
    frame_diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)
    # motion_mask는 0 또는 1의 값으로 구성된 이진 영상

    # 4. MHI 업데이트
    # OpenCV의 내장 함수를 사용하는 경우:
    # cv2.motempl.updateMotionHistory(src, dst, timestamp, duration)
    cv2.motempl.updateMotionHistory(motion_mask, mhi, timestamp, duration)

    # 직접 구현하는 경우 (간단한 디케이 적용):
    # mhi[motion_mask == 1] = timestamp
    # mhi[(motion_mask == 0) & (mhi < (timestamp - duration))] = 0

    # 5. MHI 시각화를 위해 정규화: 최근 움직임(현재 시간에 가까운 값)은 밝게 표현
    # 계산: (mhi - (timestamp - duration)) / duration -> 0~1 범위
    mhi_normalized = np.uint8(np.clip((mhi - (timestamp - duration)) / duration, 0, 1) * 255)

    # (선택 사항) MEI 생성: MHI를 이진화하여 전체 움직임 영역 확인
    _, mei = cv2.threshold(mhi_normalized, 30, 255, cv2.THRESH_BINARY)

    # 6. 결과 시각화
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Motion Mask', motion_mask * 255)  # 0/1 값을 0/255로 변환
    cv2.imshow('Motion History Image (MHI)', mhi_normalized)
    cv2.imshow('Motion Energy Image (MEI)', mei)

    # 키 입력 처리: 'q' 입력 시 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 현재 프레임을 이전 프레임으로 갱신
    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
