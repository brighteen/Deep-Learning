# -*- coding: utf-8 -*-
"""
지정된 하나의 mp4 영상을 처리하여
해당 영상과 같은 폴더 내 하위 폴더에 히트맵 결과를 저장하는 스크립트
"""

import os, cv2, numpy as np, torch, time
from ultralytics import YOLO

# 1) 처리할 단일 영상 파일 경로 (사용자 요청 경로)
file_path = r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\tile_r0_c1.mp4' # <수정 1-1> 특정 파일 경로 지정

# 2) 디바이스 설정 및 모델 로드
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = YOLO(r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt').to(device)

# 3) IoU 계산 함수 및 가중치 설정 (이 부분은 동일합니다)
def compute_iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter/union if union>0 else 0

# 참고: 사용자 설명과 달리 코드의 low_iou_weight는 1입니다.
high_iou_thresh, high_iou_weight, low_iou_weight = 0.9, 5, 1 # <코드 1-15> IoU 임계치 및 가중치 설정 (높은 IoU인 경우 가중치 5, 아니면 1)

# 4) 단일 파일 처리 시작
if not os.path.exists(file_path): # <수정 1-2> 파일이 실제로 존재하는지 확인
    print(f"[오류] 지정된 파일이 존재하지 않습니다: {file_path}")
else:
    fname = os.path.basename(file_path) # <수정 1-3> 전체 파일 이름 추출 (확장자 포함)
    video_path = file_path # <수정 1-4> 처리할 영상 경로를 지정된 파일 경로로 설정

    if not fname.lower().endswith('.mp4'): # <수정 1-5> 파일 확장자 확인
        print(f"[경고] 지정된 파일이 MP4 형식이 아닙니다: {file_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[경고] 영상 열기 실패: {video_path}")
        else:
            # 2.1) 첫 번째 프레임 저장
            ret_first, first_frame = cap.read()
            if not ret_first:
                print(f"[오류] 첫 번째 프레임 읽기 실패: {fname}")
                cap.release()
            else:
                # 다시 프레임 포지션을 처음으로 리셋
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                # 5) 영상별 출력 폴더 생성 (원본 영상과 같은 폴더 내에)
                output_base_dir = os.path.dirname(file_path) # <수정 1-6> 원본 영상이 있는 디렉터리 경로
                base = os.path.splitext(fname)[0]
                out_dir = os.path.join(output_base_dir, base) # <수정 1-7> 결과 저장 폴더 경로 설정
                os.makedirs(out_dir, exist_ok=True) # <수정 1-8> 폴더 생성

                start = time.time()
                # 6) 원본 해상도, FPS 가져오기
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 7) 리사이즈 규칙 및 샘플링 간격
                if orig_w > 1920: rw = 960
                elif orig_w > 1280: rw = 640
                else: rw = orig_w
                rh = int(orig_h * (rw/orig_w))
                sample_interval = int(fps * 5)

                heatmap, prev_boxes, first_frame, last_frame, idx = (
                    np.zeros((rh, rw), np.float32), [], first_frame, None, 0
                )
                print(f"Resizing to: {rw}×{rh}, FPS: {fps}, Sample Interval: {sample_interval}")

                # 8) 프레임 순회하며 히트맵 누적 (기존 로직과 동일)
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    idx += 1
                    if idx % sample_interval != 0: continue

                    last_frame = frame.copy()
                    small = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
                    results = model(small, device=device)[0]

                    curr = []
                    for box in results.boxes:
                        if float(box.conf.cpu()) < 0.25: continue
                        x1,y1,x2,y2 = box.xyxy.cpu().numpy().astype(int)[0]
                        curr.append([x1,y1,x2,y2])

                    for c in curr:
                        w = high_iou_weight if any(
                            compute_iou(c, p)>high_iou_thresh for p in prev_boxes
                        ) else low_iou_weight
                        x1,y1,x2,y2 = np.clip(c, 0, [rw-1, rh-1, rw-1, rh-1])
                        heatmap[y1:y2, x1:x2] += w

                    prev_boxes = curr

                cap.release()

                # 9) 샘플 프레임 없으면 건너뛰기
                if last_frame is None:
                    print(f"[오류] 샘플 프레임 없음: {fname}")
                else:
                    # 10) 후처리 및 결과 저장 (기존 로직과 동일)
                    clipv = np.percentile(heatmap, 99)
                    norm = cv2.normalize(
                        np.clip(heatmap,0,clipv), None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)
                    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                    full = cv2.resize(color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    blend = cv2.addWeighted(first_frame, 0.5, full, 0.5, 0)

                    cv2.imwrite(os.path.join(out_dir, 'heatmap_color.png'), color)
                    cv2.imwrite(os.path.join(out_dir, 'heatmap_full.png'), full)
                    cv2.imwrite(os.path.join(out_dir, 'heatmap_blended_full.png'), blend)

                    print(f"### {fname} 처리 완료 (소요 {time.time()-start:.2f}초) ###")
                    print(f"### 결과 저장 위치: {out_dir} ###")

# 전체 스크립트 실행이 완료되었음을 알림 (단일 파일 처리 후)
print("--- 단일 영상 파일 처리 스크립트 종료 ---")