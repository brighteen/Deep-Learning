import cv2
from ultralytics import YOLO
import threading
import queue
import time

# YOLO 모델 로드
model = YOLO("best_chick.pt")

video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\detect2YOLO\datas\tile_r0_c2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다:", video_path)
    exit()

# 프레임 저장용 큐 (최대 10개)
frame_queue = queue.Queue(maxsize=10)

# 프레임 읽기 스레드: 프레임을 큐에 빠르게 넣어줍니다.
def frame_reader(cap, q):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not q.full():
            q.put(frame)
    cap.release()

# 디텍션과 화면 표시 스레드
def process_frames(q):
    while True:
        if q.empty():
            time.sleep(0.01)
            continue
        frame = q.get()
        # YOLO 디텍션 수행 (요청에 따라 frame_skip 적용 가능)
        results = model.predict(frame, imgsz=1280, conf=0.3, iou=0.5)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Detection", annotated_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# 스레드 시작
reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue))
reader_thread.daemon = True
reader_thread.start()

process_frames(frame_queue)
