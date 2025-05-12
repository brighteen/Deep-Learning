# -*- coding: utf-8 -*-
"""
여러 mp4 영상을 한 번에 처리하여
영상별 하위 폴더에 히트맵 결과를 저장하는 스크립트
"""

import os, cv2, numpy as np, torch, time           # <코드 1-1> 필요한 라이브러리 임포트: 파일 시스템, OpenCV, NumPy, PyTorch, 시간 측정
from ultralytics import YOLO                       # <코드 1-2> Ultralytics YOLO 객체 탐지를 위한 모듈 임포트

# 1) 네트워크 드라이브가 로컬에 마운트된 경로 (필요에 따라 수정하세요)
video_dir   = '/Volumes/kokofarm/폐사체동영상'    # <코드 1-3> 원본 영상(.mp4)이 저장된 디렉터리 경로
output_root = '모든데이터적용'                   # <코드 1-4> 결과를 저장할 최상위 폴더 이름
os.makedirs(output_root, exist_ok=True)          # <코드 1-5> 결과 폴더가 없으면 생성

# 2) 디바이스 설정 및 모델 로드
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  
                                                  # <코드 1-6> MPS(Apple GPU) 사용 가능 시 MPS, 아니면 CPU
print(f"Using device: {device}")                 # <코드 1-7> 사용 디바이스 출력
model = YOLO('./best.pt').to(device)             # <코드 1-8> 학습된 YOLO 모델 로드 및 디바이스로 이동

# 3) IoU 계산 함수 및 가중치 설정
def compute_iou(b1, b2):                        # <코드 1-9> 두 박스 간 IoU(교집합/합집합) 계산 함수 정의
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB-xA) * max(0, yB-yA)        # <코드 1-10> 교집합 면적
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])          # <코드 1-11> 첫 번째 박스 면적
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])          # <코드 1-12> 두 번째 박스 면적
    union = area1 + area2 - inter                # <코드 1-13> 합집합 면적
    return inter/union if union>0 else 0         # <코드 1-14> IoU 값 반환

high_iou_thresh, high_iou_weight, low_iou_weight = 0.9, 5, 1  
                                                  # <코드 1-15> IoU 임계치 및 가중치 설정 (높은 IoU인 경우 가중치 5, 아니면 1)

# 4) 모든 mp4 파일 반복 처리
for fname in os.listdir(video_dir):               # <코드 1-16> 디렉터리 내 모든 파일 순회
    if not fname.lower().endswith('.mp4'):        # <코드 1-17> 확장자가 .mp4가 아니면 건너뜀
        continue

    video_path = os.path.join(video_dir, fname)   # <코드 1-18> 파일 경로 구성
    cap = cv2.VideoCapture(video_path)            # <코드 1-19> 동영상 캡처 객체 생성
    if not cap.isOpened():                        # <코드 1-20> 열기에 실패하면 경고 후 건너뜀
        print(f"[경고] 영상 열기 실패: {video_path}")
        continue

    # 2.1) 첫 번째 프레임 저장
    ret_first, first_frame = cap.read()           # <코드 1-21> 첫 프레임 읽기
    if not ret_first:                             # <코드 1-22> 읽기 실패 시 오류 메시지 출력 후 자원 해제
        print(f"[오류] 첫 번째 프레임 읽기 실패: {fname}")
        cap.release()
        continue
    # 다시 프레임 포지션을 처음으로 리셋
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # <코드 1-23> 프레임 인덱스를 0으로 재설정

    # 5) 영상별 출력 폴더 생성
    base = os.path.splitext(fname)[0]             # <코드 1-24> 확장자를 제외한 파일명 추출
    out_dir = os.path.join(output_root, base)     # <코드 1-25> 영상별 하위 폴더 경로
    os.makedirs(out_dir, exist_ok=True)           # <코드 1-26> 폴더 생성

    start = time.time()                           # <코드 1-27> 처리 시간 측정 시작
    # 6) 원본 해상도, FPS 가져오기
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # <코드 1-28> 원본 영상 너비
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # <코드 1-29> 원본 영상 높이
    fps    = cap.get(cv2.CAP_PROP_FPS)               # <코드 1-30> 초당 프레임 수

    # 7) 리사이즈 규칙 및 샘플링 간격
    if orig_w > 1920: rw = 960                    # <코드 1-31> 가로 해상도 >1920 → 960으로 축소
    elif orig_w > 1280: rw = 640                  # <코드 1-32> >1280 → 640으로 축소
    else: rw = orig_w                             # <코드 1-33> 그 외는 원본 유지
    rh = int(orig_h * (rw/orig_w))                # <코드 1-34> 비율에 맞춰 세로 해상도 계산
    sample_interval = int(fps * 5)                # <코드 1-35> 5초마다 한 프레임 처리

    heatmap, prev_boxes, first_frame, last_frame, idx = (
        np.zeros((rh, rw), np.float32), [], first_frame, None, 0
    )                                              # <코드 1-36> 히트맵 배열·이전 박스·첫/마지막 프레임·인덱스 초기화
    print(f"Resizing to: {rw}×{rh}, FPS: {fps}, Sample Interval: {sample_interval}")

    # 8) 프레임 순회하며 히트맵 누적
    while True:
        ret, frame = cap.read()                   # <코드 1-37> 다음 프레임 읽기
        if not ret: break                         # <코드 1-38> 영상 끝나면 루프 탈출
        idx += 1
        if idx % sample_interval != 0: continue   # <코드 1-39> 샘플링 간격이 아니면 건너뜀

        last_frame = frame.copy()                 # <코드 1-40> 마지막 샘플 프레임 저장
        small = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)  
                                                  # <코드 1-41> 축소된 크기로 리사이즈
        results = model(small, device=device)[0]  # <코드 1-42> YOLO 객체 탐지 실행

        curr = []                                 # <코드 1-43> 현재 프레임 박스 리스트 초기화
        for box in results.boxes:                 
            if float(box.conf.cpu()) < 0.25: continue  
                                                  # <코드 1-44> 신뢰도 0.25 미만 박스 필터링
            x1,y1,x2,y2 = box.xyxy.cpu().numpy().astype(int)[0]  
                                                  # <코드 1-45> 좌표 정수로 변환
            curr.append([x1,y1,x2,y2])            # <코드 1-46> 현재 박스 목록에 추가

        for c in curr:                            # <코드 1-47> 각 박스별 가중치 계산 및 히트맵 누적
            w = high_iou_weight if any(
                compute_iou(c, p)>high_iou_thresh for p in prev_boxes
            ) else low_iou_weight
            x1,y1,x2,y2 = np.clip(c, 0, [rw-1, rh-1, rw-1, rh-1])  
                                                  # <코드 1-48> 히트맵 범위 내 좌표 정리
            heatmap[y1:y2, x1:x2] += w            # <코드 1-49> 가중치만큼 히트맵 값 증가

        prev_boxes = curr                         # <코드 1-50> 이번 프레임 박스를 다음 비교용으로 저장

    cap.release()                                # <코드 1-51> 자원 해제

    # 9) 샘플 프레임 없으면 건너뛰기
    if last_frame is None:                       
        print(f"[오류] 샘플 프레임 없음: {fname}")
        continue

    # 10) 후처리 및 결과 저장
    clipv = np.percentile(heatmap, 99)            # <코드 1-52> 히트맵 상위 1% 값 계산
    norm  = cv2.normalize(
        np.clip(heatmap,0,clipv), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)                            # <코드 1-53> 0~255 범위로 정규화 및 uint8 변환
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)  
                                                  # <코드 1-54> JET 컬러맵 적용
    full  = cv2.resize(color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)  
                                                  # <코드 1-55> 원본 크기로 다시 확장
    blend = cv2.addWeighted(first_frame, 0.5, full, 0.5, 0)  
                                                  # <코드 1-56> 첫 프레임과 히트맵 블렌딩

    cv2.imwrite(os.path.join(out_dir, 'heatmap_color.png'),      color)  
                                                  # <코드 1-57> 컬러맵 이미지 저장
    cv2.imwrite(os.path.join(out_dir, 'heatmap_full.png'),       full)   
                                                  # <코드 1-58> 전체 해상도 히트맵 저장
    cv2.imwrite(os.path.join(out_dir, 'heatmap_blended_full.png'), blend)  
                                                  # <코드 1-59> 블렌드 이미지 저장

    print(f"### {fname} 처리 완료 (소요 {time.time()-start:.2f}초) ###")  
                                                  # <코드 1-60> 처리 완료 로그