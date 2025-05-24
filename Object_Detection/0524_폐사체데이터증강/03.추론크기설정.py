import cv2
import os
from ultralytics import YOLO  # YOLOv8 라이브러리

def detect_single_frame_with_custom_size(video_path, model_path, output_path, frame_number=0, img_size=(2880, 1620)):
    """
    영상에서 특정 프레임을 추출하여 지정된 크기로 YOLO 객체 탐지 수행
    """
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("영상을 열 수 없습니다")
        return None
    
    # 전체 프레임 수 확인
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 프레임 번호 확인 및 조정
    if frame_number >= total_frames:
        frame_number = 0
        print(f"요청한 프레임이 총 프레임 수를 초과하여 첫 번째 프레임을 사용합니다.")
    
    # 해당 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("프레임을 읽을 수 없습니다")
        cap.release()
        return None
    
    # 모델 로드
    print(f"모델 로딩 중: {model_path}")
    model = YOLO(model_path)
    
    # 지정된 크기로 객체 탐지 수행
    print(f"객체 탐지 중... (이미지 크기: {img_size})")
    results = model(frame, imgsz=img_size)
    
    # 결과 시각화
    annotated_frame = results[0].plot()
    
    # 정보 추가
    cv2.putText(annotated_frame, f"이미지 크기: {img_size[0]}x{img_size[1]}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 탐지된 객체 수
    num_objects = len(results[0].boxes)
    cv2.putText(annotated_frame, f"탐지된 객체: {num_objects}개", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 결과 저장
    cv2.imwrite(output_path, annotated_frame)
    
    # 자원 해제
    cap.release()
    
    print(f"탐지 결과가 {output_path}에 저장되었습니다")
    print(f"탐지된 객체 수: {num_objects}")
    return output_path

# 경로 설정
video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4"
model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best.pt"
output_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\detected_frame_2880x1620.jpg"

# 100번째 프레임으로 객체 탐지 수행
detect_single_frame_with_custom_size(
    video_path=video_path,
    model_path=model_path,
    output_path=output_path,
    frame_number=100,
    img_size=(2880, 1620)  # 요청한 크기로 설정
)