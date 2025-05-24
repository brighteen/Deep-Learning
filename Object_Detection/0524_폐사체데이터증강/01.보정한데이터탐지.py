import cv2
import torch
from ultralytics import YOLO  # YOLOv8 라이브러리
import os
import time

def detect_objects_in_video(model_path, video_path, output_path):
    print(f"모델 로딩 중: {model_path}")
    model = YOLO(model_path)  # YOLOv8 모델 로드
    
    print(f"영상 로딩 중: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return
    
    # 영상 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 출력 비디오 설정
    output_file = os.path.join(output_path, "탐지결과.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOv8로 객체 탐지
        results = model(frame)
        
        # 결과를 프레임에 시각화
        annotated_frame = results[0].plot()
        
        # 프레임에 정보 추가
        cv2.putText(annotated_frame, f"프레임: {frame_count+1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 탐지된 객체 정보
        detections = results[0].boxes
        num_objects = len(detections)
        cv2.putText(annotated_frame, f"탐지된 객체: {num_objects}개", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 결과 저장
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            print(f"처리 중: {frame_count}/{total_frames} 프레임 (처리 속도: {fps_processing:.2f} fps)")
    
    cap.release()
    out.release()
    
    print(f"처리 완료: {frame_count} 프레임 처리됨")
    print(f"결과 저장 경로: {output_file}")
    
    return output_file

# 경로 설정
model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_yolo11.pt"
video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\왜곡보정\첫5초_왜곡보정.mp4"
output_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\탐지결과"

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 객체 탐지 실행
detect_objects_in_video(model_path, video_path, output_dir)