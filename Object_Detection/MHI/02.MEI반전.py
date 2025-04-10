import cv2
import numpy as np

# 1. 파라미터 설정 및 라이브러리 초기화
video_path = r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\tile_r0_c1.mp4'  # 혹은 카메라의 경우 0
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    exit()

# MHI를 위한 파라미터 (duration: MHI가 유지될 시간)
duration = 1.0  # 초 단위
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

# 2. 프레임 반복 처리
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 타임스탬프 업데이트
    timestamp += dt

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. 프레임 차분을 이용한 움직임 검출
    frame_diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)
    # motion_mask는 0 또는 1의 값 (이진 영상)

    # 4. MHI 업데이트 (OpenCV 내장 함수 이용)
    cv2.motempl.updateMotionHistory(motion_mask, mhi, timestamp, duration)

    # 5. MHI 정규화: (mhi - (timestamp - duration)) / duration 범위를 0~1로 정규화 후 0~255로 스케일링
    mhi_normalized = np.uint8(np.clip((mhi - (timestamp - duration)) / duration, 0, 1) * 255)

    # 6. MEI 생성: MHI를 일정 임계값으로 이진화
    _, mei = cv2.threshold(mhi_normalized, 30, 255, cv2.THRESH_BINARY)
    
    # 7. MEI의 반전 이미지 생성
    inverted_mei = cv2.bitwise_not(mei)  # 또는 inverted_mei = 255 - mei

    # 8. 결과 시각화: 각 창에 원본 프레임, 움직임 마스크, MHI, MEI, 반전 MEI 출력
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Motion Mask', motion_mask * 255)  # 0/1 값을 0/255로 변환하여 출력
    cv2.imshow('Motion History Image (MHI)', mhi_normalized)
    cv2.imshow('Motion Energy Image (MEI)', mei)
    cv2.imshow('Inverted MEI', inverted_mei)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 현재 프레임을 이전 프레임으로 갱신
    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
