import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
from torchvision.ops import nms  # torchvision의 NMS 함수 사용

def tile_based_detection(image, model, tile_size=640, overlap=0.2):
    """
    이미지를 타일로 나누어 각각 객체 탐지 수행 후 결과 합치기
    """
    h, w = image.shape[:2]
    all_predictions = []
    
    # 타일 간 겹침 계산
    stride = int(tile_size * (1 - overlap))
    
    # 진행 상황 계산을 위한 변수
    total_tiles = ((h - 1) // stride + 1) * ((w - 1) // stride + 1)
    processed_tiles = 0
    
    print(f"이미지 크기: {w}x{h}, 타일 크기: {tile_size}, 겹침: {overlap*100}%")
    print(f"총 타일 수: {total_tiles}")
    
    # 이미지를 타일로 나누어 각각 탐지 수행
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 타일 경계 계산
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            
            # 마지막 타일이 너무 작으면 조정
            if x2 - x < tile_size * 0.5:
                x = max(0, x2 - tile_size)
            if y2 - y < tile_size * 0.5:
                y = max(0, y2 - tile_size)
            
            # 타일 추출
            tile = image[y:y2, x:x2]
            
            # 타일이 비어있거나 너무 작으면 건너뛰기
            if tile.size == 0 or tile.shape[0] < 32 or tile.shape[1] < 32:
                processed_tiles += 1
                continue
            
            # 타일에서 객체 탐지 수행
            results = model(tile)
            
            # 결과가 있으면 좌표 조정하여 저장
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # 박스 좌표 조정 (x, y 오프셋 적용)
                    box_data = box.data.clone()
                    box_data[0, 0:4:2] += x  # x 좌표 조정
                    box_data[0, 1:4:2] += y  # y 좌표 조정
                    all_predictions.append(box_data)
            
            processed_tiles += 1
            if processed_tiles % 10 == 0 or processed_tiles == total_tiles:
                print(f"처리 중: {processed_tiles}/{total_tiles} 타일 완료")
    
    # 결과가 없으면 빈 배열 반환
    if not all_predictions:
        return None
    
    # 결과 합치기
    all_boxes = torch.cat(all_predictions, dim=0)
    
    # NMS(Non-Maximum Suppression) 적용하여 중복 제거
    boxes = all_boxes[:, :4]
    scores = all_boxes[:, 4]
    labels = all_boxes[:, 5]
    
    # torchvision의 NMS 적용
    iou_threshold = 0.45
    keep_indices = []
    
    # 클래스별로 NMS 적용
    for cls in torch.unique(labels):
        cls_mask = (labels == cls)
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        # NMS 적용
        cls_keep = nms(cls_boxes, cls_scores, iou_threshold)
        
        # 원래 인덱스로 변환하여 보존
        cls_indices = torch.nonzero(cls_mask).squeeze(1)
        keep_indices.append(cls_indices[cls_keep])
    
    # 모든 클래스의 결과 합치기
    if keep_indices:
        keep = torch.cat(keep_indices)
        return all_boxes[keep]
    else:
        return None

def detect_video_frame_with_tiles(video_path, model_path, output_path, frame_number=100, tile_size=640, overlap=0.2):
    """
    비디오 프레임에 타일 기반 객체 탐지 적용
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
    
    # 타일 기반 객체 탐지 수행
    print(f"타일 기반 객체 탐지 시작 (타일 크기: {tile_size}, 겹침: {overlap*100}%)")
    results = tile_based_detection(frame, model, tile_size, overlap)
    
    # 결과 시각화
    annotated_frame = frame.copy()
    
    if results is not None:
        # 각 박스 그리기
        for box in results:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = float(box[4])
            cls_id = int(box[5])
            
            # 클래스 이름 (모델에서 가져오기)
            class_names = model.names
            label = f"{class_names[cls_id]} {conf:.2f}"
            
            # 박스와 레이블 그리기
            color = (0, 255, 0)  # 초록색
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 정보 추가
    cv2.putText(annotated_frame, f"타일 크기: {tile_size}x{tile_size}, 겹침: {overlap*100}%", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 탐지된 객체 수
    num_objects = len(results) if results is not None else 0
    cv2.putText(annotated_frame, f"탐지된 객체: {num_objects}개", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 결과 저장
    cv2.imwrite(output_path, annotated_frame)
    
    # 자원 해제
    cap.release()
    
    print(f"타일 기반 객체 탐지 완료")
    print(f"탐지 결과가 {output_path}에 저장되었습니다")
    print(f"탐지된 객체 수: {num_objects}")
    
    return output_path

# 경로 설정
video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4"
model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best.pt"
output_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\tile_based_detection.jpg"

# 타일 기반 객체 탐지 실행
detect_video_frame_with_tiles(
    video_path=video_path,
    model_path=model_path,
    output_path=output_path,
    frame_number=100,
    tile_size=640,  # 타일 크기
    overlap=0.2     # 타일 간 겹침 비율 (20%)
)