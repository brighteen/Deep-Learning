import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os

# 모델 로드
model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best.pt"
model = YOLO(model_path)

# 비디오 로드
video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("오류: 영상을 열 수 없습니다.")
    exit()

# 영상 속성 확인
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"비디오 FPS: {fps}")

# 관심 영역 설정
roi_y1, roi_y2 = 800, 1600  # 세로 범위
roi_x1, roi_x2 = 700, 1800  # 가로 범위

# 지정된 초 위치의 프레임으로 변환
frame_times = [324.2, 326.4, 327.4, 329.0]  # 초 단위
frame_positions = [int(t * fps) for t in frame_times]

# 결과 저장할 디렉토리 설정
# output_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\0528_id집합표현개선2\두 프레임비교"
output_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\0609중심점\results"
os.makedirs(output_dir, exist_ok=True)

# 색상 설정 (B,G,R)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# 프레임 처리 함수
def process_frame(frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        print(f"프레임 {frame_idx} 읽기 실패")
        return None, None, None
    
    # ROI 적용
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    
    # YOLO로 객체 탐지
    results = model.track(roi, persist=True)  # persist=True로 추적 활성화
    
    # 결과 이미지에 박스와 ID 그리기
    annotated_roi = roi.copy()
    detected_ids = []
    boxes_info = []  # 박스와 ID 정보 저장
    
    if results and len(results) > 0:
        boxes = results[0].boxes
        if hasattr(boxes, 'id') and boxes.id is not None:
            for i, (box, id) in enumerate(zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().numpy())):
                # 클래스 확인 (chick만 처리)
                cls = int(boxes.cls[i].item())
                cls_name = model.names[cls]

                # chick 클래스만 처리
                if cls_name.lower() != 'chick':
                    continue

                id = int(id)
                x1, y1, x2, y2 = box.astype(int)

                # 박스 그리기 (모두 초록색)
                cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), GREEN, 2)

                # 중심점 계산 및 점 찍기
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(annotated_roi, (center_x, center_y), 5, (0, 255, 255), -1)  # 노란색 점

                # ID 표시 (한글 없이)
                cv2.putText(annotated_roi, f"ID: {id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)

                detected_ids.append(id)
                boxes_info.append({
                    'id': id,
                    'box': (x1, y1, x2, y2),
                    'center': (center_x, center_y)
                })
    
    # 프레임 정보 추가 (한글 없이)
    cv2.putText(annotated_roi, f"Frame: {frame_idx}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 감지된 ID 정보 추가 (한글 없이)
    id_text = f"Detected IDs: {len(detected_ids)}"
    cv2.putText(annotated_roi, id_text, (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return annotated_roi, detected_ids, boxes_info

# 모든 프레임 처리 및 저장
frames_info = []

for i, frame_idx in enumerate(frame_positions):
    print(f"프레임 {frame_idx} (시간: {frame_times[i]}초) 처리 중...")
    roi, ids, boxes = process_frame(frame_idx)
    
    if roi is not None:
        # 관심 영역만 저장
        output_path = os.path.join(output_dir, f"frame_{frame_idx}_time_{frame_times[i]:.1f}s_roi.jpg")
        cv2.imwrite(output_path, roi)
        print(f"관심 영역 저장됨: {output_path}")
        
        frames_info.append({
            "frame_idx": frame_idx,
            "time": frame_times[i],
            "ids": ids,
            "boxes": boxes,
            "roi": roi.copy()
        })

# 비디오 리소스 해제
cap.release()

# 각 연속된 두 프레임 간의 비교
for i in range(1, len(frames_info)):
    prev_frame = frames_info[i-1]
    curr_frame = frames_info[i]
    
    comparison_path = os.path.join(output_dir, f"id_comparison_{prev_frame['frame_idx']}_{curr_frame['frame_idx']}.txt")
    with open(comparison_path, "w", encoding='utf-8') as f:
        f.write(f"프레임 {prev_frame['frame_idx']} (시간: {prev_frame['time']:.1f}초) ID 목록: {prev_frame['ids']}\n")
        f.write(f"프레임 {curr_frame['frame_idx']} (시간: {curr_frame['time']:.1f}초) ID 목록: {curr_frame['ids']}\n")
        f.write("\n비교 분석:\n")
        
        # ID 집합 간 교집합 (두 프레임 모두에 있는 ID)
        common_ids = set(prev_frame['ids']) & set(curr_frame['ids'])
        f.write(f"양쪽 프레임에 모두 있는 ID: {common_ids}\n")
        f.write(f"양쪽 프레임에 공통 ID 개수: {len(common_ids)}\n")
        
        # 첫 번째 프레임에만 있는 ID
        only_in_prev = set(prev_frame['ids']) - set(curr_frame['ids'])
        f.write(f"이전 프레임에만 있는 ID: {only_in_prev}\n")
        f.write(f"이전 프레임에만 있는 ID 개수: {len(only_in_prev)}\n")
        
        # 두 번째 프레임에만 있는 ID
        only_in_curr = set(curr_frame['ids']) - set(prev_frame['ids'])
        f.write(f"현재 프레임에만 있는 ID: {only_in_curr}\n")
        f.write(f"현재 프레임에만 있는 ID 개수: {len(only_in_curr)}\n")
    
    print(f"ID 비교 결과 저장됨: {comparison_path}")
    
    # 비교 이미지 생성 (공통 ID는 표시하지 않음)
    comparison_image = curr_frame['roi'].copy()
    
    # 이전 프레임에만 있던 객체(사라진 객체)는 빨간색으로 표시
    for box_info in prev_frame['boxes']:
        if box_info['id'] in only_in_prev:
            x1, y1, x2, y2 = box_info['box']
            cv2.rectangle(comparison_image, (x1, y1), (x2, y2), RED, 2)
            cv2.putText(comparison_image, f"ID: {box_info['id']} (Lost)", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
    
    # 현재 프레임에만 있는 객체(새로 나타난 객체)는 파란색으로 표시
    for box_info in curr_frame['boxes']:
        if box_info['id'] in only_in_curr:
            x1, y1, x2, y2 = box_info['box']
            cv2.rectangle(comparison_image, (x1, y1), (x2, y2), BLUE, 2)
            cv2.putText(comparison_image, f"ID: {box_info['id']} (New)", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLUE, 2)
    
    # 비교 이미지 저장
    comparison_path = os.path.join(output_dir, f"id_changes_{prev_frame['frame_idx']}_{curr_frame['frame_idx']}.jpg")
    cv2.imwrite(comparison_path, comparison_image)
    print(f"ID 변화 비교 이미지 저장됨: {comparison_path}")

print("처리 완료!")