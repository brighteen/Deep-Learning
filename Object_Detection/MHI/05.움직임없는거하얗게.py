import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt")
cap = cv2.VideoCapture(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\tile_r0_c1.mp4")

ret, prev_frame = cap.read()
if not ret:
    print("초기 프레임 로딩 실패")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)
lower_white = np.array([0, 0, 150])
upper_white = np.array([180, 60, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray, prev_gray)
    frame_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)

    _, motion_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, np.ones((5, 5), np.uint8), iterations=1)

    results = model(frame)
    annotated_frame = results[0].plot()
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()  # 클래스 ID 배열

    # 전체 chick_mask 초기화 (검은 배경)
    chick_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # 각 검출된 닭 영역에 대해 처리
    for box, cls_id in zip(boxes, classes):
        if int(cls_id) != 0:  # 0번 클래스가 닭으로 가정
            continue

        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

        # 해당 박스 영역에서 HSV를 이용해 닭 영역 마스크 생성
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_roi = cv2.inRange(hsv, lower_white, upper_white)

        # 해당 박스 영역의 모션 정보 추출
        roi_motion = motion_mask[y1:y2, x1:x2]
        motion_ratio = np.count_nonzero(roi_motion) / (roi_motion.size + 1e-6)

        # 움직임이 적은 경우(폐사체로 판단, motion_ratio < 0.02)만 흰색 처리
        if motion_ratio < 0.02:
            chick_mask[y1:y2, x1:x2] = cv2.bitwise_or(chick_mask[y1:y2, x1:x2], mask_roi)
        else:
            # 움직임이 많으면 해당 영역은 검은색(0)으로 유지함
            chick_mask[y1:y2, x1:x2] = 0

    cv2.imshow("YOLO Detection", annotated_frame)
    cv2.imshow("Chick Mask (All)", chick_mask)

    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
