# -*- coding: utf-8 -*-
"""
지정된 하나의 mp4 영상을 처리하여
해당 영상과 같은 폴더 내 하위 폴더에 히트맵 결과를 저장하는 스크립트

YOLO 탐지 결과 중 '닭' 클래스만 사용하여 히트맵을 생성합니다.
"""

import os, cv2, numpy as np, torch, time
from ultralytics import YOLO

# 1) 처리할 단일 영상 파일 경로 (사용자 요청 경로)
# TODO: 이곳의 파일 경로를 실제 처리할 MP4 영상 파일 경로로 수정하세요.
file_path = r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4'

# 2) 디바이스 설정 및 모델 로드
# MPS (Apple Silicon GPU) 사용 가능 시 MPS, 아니면 CPU 사용
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# TODO: 학습된 YOLO 모델 파일 경로를 확인하세요. 보통 'best.pt'입니다.
try:
    model = YOLO(r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt').to(device)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"[오류] YOLO 모델 로드 실패: {e}")
    print("모델 파일('best.pt') 경로를 확인하거나 파일이 존재하는지 확인하세요.")
    exit() # 모델 로드 실패 시 스크립트 종료

# 3) IoU 계산 함수 및 가중치 설정
def compute_iou(b1, b2):
    """
    두 개의 바운딩 박스 (x1, y1, x2, y2 형식) 간의 Intersection over Union (IoU)를 계산합니다.
    """
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB-xA) * max(0, yB-yA) # 교집합 면적
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1]) # 첫 번째 박스 면적
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1]) # 두 번째 박스 면적
    union = area1 + area2 - inter # 합집합 면적
    return inter/union if union>0 else 0 # IoU 값 반환 (union이 0이면 0 반환)

# IoU 임계치 및 가중치 설정:
# high_iou_thresh: 이 값 이상이면 '높은 겹침'으로 판단
# high_iou_weight: 높은 겹침을 가진 박스가 히트맵에 더하는 가중치
# low_iou_weight: 낮은 겹침을 가진 박스가 히트맵에 더하는 가중치
# (참고: low_iou_weight는 현재 1입니다. 움직임이 둔한 닭과 폐사체를 더 잘 구분하려면 이 값을 1보다 훨씬 작은 양수 값으로 실험해볼 수 있습니다. 예: 0.1 또는 0.5)
high_iou_thresh = 0.9
high_iou_weight = 5
low_iou_weight = 1 # 이 값을 조정하여 실험해보세요.

print(f"IoU Settings: Threshold={high_iou_thresh}, High Weight={high_iou_weight}, Low Weight={low_iou_weight}")


# 4) 단일 파일 처리 시작
if not os.path.exists(file_path):
    print(f"[오류] 지정된 파일이 존재하지 않습니다: {file_path}")
elif not file_path.lower().endswith('.mp4'):
    print(f"[경고] 지정된 파일이 MP4 형식이 아닙니다: {file_path}")
else:
    fname = os.path.basename(file_path) # 전체 파일 이름 추출 (확장자 포함)
    video_path = file_path # 처리할 영상 경로 지정

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[경고] 영상 열기 실패: {video_path}")
    else:
        # 2.1) 첫 번째 프레임 저장 (나중에 블렌딩에 사용)
        ret_first, first_frame = cap.read()
        if not ret_first:
            print(f"[오류] 첫 번째 프레임 읽기 실패: {fname}")
            cap.release()
        else:
            # 다시 프레임 포지션을 처음으로 리셋 (메인 처리 시작 전)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # 5) 영상별 출력 폴더 생성 (원본 영상과 같은 폴더 내에)
            output_base_dir = os.path.dirname(file_path) # 원본 영상이 있는 디렉터리 경로
            base = os.path.splitext(fname)[0] # 확장자를 제외한 파일 이름
            out_dir = os.path.join(output_base_dir, base) # 결과 저장 폴더 경로 설정
            os.makedirs(out_dir, exist_ok=True) # 폴더가 없으면 생성

            start = time.time() # 처리 시간 측정 시작

            # 6) 원본 해상도, FPS 가져오기
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 7) 리사이즈 규칙 및 샘플링 간격
            # YOLO 모델 입력 크기를 줄여 처리 속도 개선
            if orig_w > 1920: rw = 960
            elif orig_w > 1280: rw = 640
            else: rw = orig_w
            rh = int(orig_h * (rw/orig_w)) # 비율에 맞춰 세로 해상도 계산

            # 5초마다 한 프레임 처리 (샘플링 간격)
            # TODO: 필요시 샘플링 간격 (초) 조절 가능
            sample_interval_seconds = 5
            sample_interval = max(1, int(fps * sample_interval_seconds)) # 0이 되지 않도록 최소 1

            print(f"Original Resolution: {orig_w}x{orig_h}, Resizing to: {rw}x{rh}, FPS: {fps:.2f}, Sample Interval: {sample_interval} frames ({sample_interval_seconds} seconds)")

            # 히트맵 배열, 이전 프레임의 박스 목록, 마지막 샘플 프레임, 프레임 인덱스 초기화
            # 히트맵은 리사이즈된 크기로 생성
            heatmap = np.zeros((rh, rw), np.float32)
            prev_boxes = []
            last_frame = None
            idx = 0 # 프레임 인덱스

            # 8) 프레임 순회하며 '닭' 객체 위치에 히트맵 누적
            print(f"Processing video: {fname}")
            while True:
                ret, frame = cap.read() # 다음 프레임 읽기
                if not ret:
                    break # 영상 끝나면 루프 탈출

                idx += 1
                # 설정된 샘플링 간격이 아니면 건너뜀
                if idx % sample_interval != 0:
                    continue

                # 현재 프레임이 샘플 프레임인 경우 저장
                last_frame = frame.copy()

                # 모델 입력 크기로 리사이즈
                small_frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)

                # YOLO 객체 탐지 실행
                results = model(small_frame, device=device, verbose=False)[0] # verbose=False로 추론 과정 출력을 줄임

                curr_chicken_boxes = [] # 현재 프레임에서 탐지된 '닭' 박스 목록

                # --- 여기에 '닭' 클래스 ID를 설정합니다. ---
                # TODO: !!!! 중요 !!!!
                #       사용하신 YOLO 모델의 '닭' 클래스 ID를 정확히 입력하세요.
                #       dataset.yaml 파일을 확인하거나, 이전 답변의 방법으로 ID를 찾으세요.
                # 예시: '닭' 클래스 ID가 0번일 경우:
                chicken_class_id = 0 # <--- 이곳을 실제 '닭' 클래스 ID로 변경하세요!
                # ------------------------------------------

                for box in results.boxes:
                    confidence = float(box.conf.cpu().numpy())
                    class_id = int(box.cls.cpu().numpy())

                    # 1. 신뢰도(confidence) 임계치 확인
                    if confidence < 0.25: # TODO: 필요시 신뢰도 임계치 조정 가능
                        continue # 신뢰도가 낮으면 건너뜀

                    # 2. 객체 클래스 ID 확인 - '닭' 클래스만 선택
                    if class_id != chicken_class_id:
                        continue # '닭' 클래스가 아니면 건너뜀

                    # 신뢰도 통과 및 '닭' 클래스인 경우에만 좌표 추출
                    x1,y1,x2,y2 = box.xyxy.cpu().numpy().astype(int)[0] # 좌표 정수로 변환

                    # 현재 프레임의 '닭' 박스 목록에 추가
                    curr_chicken_boxes.append([x1,y1,x2,y2])

                # 현재 프레임에서 탐지된 각 '닭' 박스에 대해 히트맵 누적
                for c in curr_chicken_boxes:
                    # 현재 '닭' 박스가 이전 프레임의 '닭' 박스들과 높은 IoU를 가지는지 확인
                    # 이전 프레임의 박스도 '닭' 박스 목록(prev_boxes)만 사용하고 있습니다.
                    has_high_overlap = any(compute_iou(c, p) > high_iou_thresh for p in prev_boxes)

                    # 겹침 정도에 따라 가중치 설정
                    w = high_iou_weight if has_high_overlap else low_iou_weight

                    # 히트맵 배열 범위 내 좌표 정리
                    x1,y1,x2,y2 = np.clip(c, 0, [rw-1, rh-1, rw-1, rh-1])

                    # 해당 박스 영역에 가중치만큼 히트맵 값 증가 (누적)
                    heatmap[y1:y2, x1:x2] += w

                # 이번 프레임의 '닭' 박스 목록을 다음 비교용으로 저장
                prev_boxes = curr_chicken_boxes

            # 영상 처리 루프 종료 후 자원 해제
            cap.release()

            # 9) 샘플 프레임이 하나도 없었으면 (매우 짧은 영상 등) 건너뛰기
            if last_frame is None:
                print(f"[오류] 처리할 샘플 프레임이 없습니다 (영상이 너무 짧거나 FPS가 낮을 수 있습니다): {fname}")
            else:
                # 10) 후처리 및 결과 저장
                # 히트맵 값의 상위 1% 값을 기준으로 클리핑 (이상치 영향 감소)
                clipv = np.percentile(heatmap, 99) # TODO: 필요시 퍼센타일 값 조정 가능

                # 값을 0~clipv 범위로 제한 후 0~255 범위로 정규화
                norm = cv2.normalize(
                    np.clip(heatmap, 0, clipv), None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8) # uint8 타입으로 변환 (이미지 저장을 위함)

                # JET 컬러맵 적용
                color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

                # 히트맵 이미지를 원본 영상 크기로 다시 확장
                full = cv2.resize(color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

                # 원본 영상의 첫 프레임과 확장된 히트맵 이미지를 블렌딩
                blend = cv2.addWeighted(first_frame, 0.5, full, 0.5, 0) # TODO: 블렌딩 비율 (0.5, 0.5) 조정 가능

                # 결과 이미지 저장
                cv2.imwrite(os.path.join(out_dir, 'heatmap_color_resized.png'), color) # 리사이즈된 크기
                cv2.imwrite(os.path.join(out_dir, 'heatmap_full.png'), full) # 원본 크기 컬러맵
                cv2.imwrite(os.path.join(out_dir, 'heatmap_blended_full.png'), blend) # 원본 크기 블렌딩

                print(f"### {fname} 처리 완료 (소요 {time.time()-start:.2f}초) ###")
                print(f"### 결과 저장 위치: {out_dir} ###")

# 전체 스크립트 실행이 완료되었음을 알림
print("--- 단일 영상 파일 처리 스크립트 종료 ---")