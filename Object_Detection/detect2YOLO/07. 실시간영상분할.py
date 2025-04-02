import cv2
from ultralytics import YOLO
import time

# 모델 로드 (기존 코드와 동일)
model = YOLO("best_chick.pt")

# 비디오 파일 열기 (기존 코드와 동일)
cap = cv2.VideoCapture(r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\detect2YOLO\datas\0_8_IPC1_20221105100432.mp4")

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
segment_duration = 10  # 초
frames_per_segment = frame_rate * segment_duration
num_segments = (total_frames + frames_per_segment - 1) // frames_per_segment # 총 분할 개수 계산

for i in range(num_segments):
    start_frame = i * frames_per_segment
    end_frame = min((i + 1) * frames_per_segment, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # 현재 프레임 위치 설정
    segment_frames = []
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        segment_frames.append(frame)

    print(f"분할 {i+1} (프레임 {start_frame} ~ {end_frame-1}) 재생 시작...")

    for frame in segment_frames:
        # 여기에 각 프레임에 대한 객체 감지 및 추적 로직 (기존 while 루프 내부 코드)을 적용합니다.
        # results = model.predict(frame, imgsz=1280, conf=0.3, iou=0.5)
        # ... (기존 추적 및 시각화 코드) ...

        cv2.imshow("Segmented Video", frame) # 분할된 영상 보여주기
        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):
            break

    print(f"분할 {i+1} 재생 완료. 다음 분할까지 10초 대기...")
    time.sleep(10)

cap.release()
cv2.destroyAllWindows()