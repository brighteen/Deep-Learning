import cv2
import numpy as np

# 1. 동영상 입력 및 배경 제거기 초기화
video_path = r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\tile_r0_c3.mp4'  # 혹은 0 (카메라)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

# OpenCV의 MOG2 배경 제거기 초기화 (detectShadows=False로 설정하여 그림자 제거)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)

# 프레임 차분을 위한 초기 전 프레임 준비 (그레이스케일)
ret, frame = cap.read()
if not ret:
    print("첫 번째 프레임을 읽지 못했습니다.")
    cap.release()
    exit()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 후처리용 커널 (morphological 연산)
kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 배경 제거: foreground mask (객체 영역: 255, 배경: 0)
    fg_mask = bg_subtractor.apply(frame)
    # (필요하면 추가 전처리: 모폴로지 연산으로 노이즈 제거)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # 3. 프레임 차분을 통한 움직임 검출 (움직임 mask)
    frame_diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    # (필요하면 모폴로지 연산 적용)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

    # 4. 움직임이 없는 객체 영역 추출
    # motion_mask_inv: 움직임이 없는 영역은 255, 움직임이 있으면 0
    motion_mask_inv = cv2.bitwise_not(motion_mask)
    # 정지한 객체 = foreground (모든 객체) AND 움직임 없는 영역
    static_objects = cv2.bitwise_and(fg_mask, motion_mask_inv)
    
    # 최종 결과: 배경는 검정(0), 움직이는 객체는 motion_mask에 의해 제거되어 검정, 정지한 객체는 흰색(255)
    # static_objects 결과를 후처리하여 노이즈 제거할 수 있음.
    static_objects = cv2.morphologyEx(static_objects, cv2.MORPH_OPEN, kernel)

    # 5. 결과 창 출력: 원본, foreground mask, motion mask, static_objects 결과
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Motion Mask', motion_mask)
    cv2.imshow('Static Objects', static_objects)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 현재 프레임을 이전 프레임으로 갱신
    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
