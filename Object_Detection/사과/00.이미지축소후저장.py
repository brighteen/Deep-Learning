import os
import cv2
import numpy as np

# 설정
base_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\사과\images"
subdirs = ["normal", "abnormal"]
scale = 0.1  # 10% 크기로 축소

# 처리할 확장자 필터
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for sub in subdirs:
    folder = os.path.join(base_dir, sub)
    if not os.path.isdir(folder):
        print(f"[경고] 폴더가 없습니다: {folder}")
        continue

    for fname in os.listdir(folder):
        if not fname.lower().endswith(EXTS):
            continue

        path = os.path.join(folder, fname)
        # --- 파일 읽기 (한글 경로 우회) ---
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[오류] 이미지를 읽을 수 없음: {path}")
            continue

        # --- 리사이즈 ---
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # --- 저장 ---
        # 1) 원본 덮어쓰기
        # save_path = path

        # 2) 원본 보존 + 새 파일명 (예: r01.jpg → r01_resized.jpg)
        name, ext = os.path.splitext(fname)
        save_fname = f"{name}_resized{ext}"
        save_path = os.path.join(folder, save_fname)

        # imencode + tofile 로 저장 (한글 경로 대응)
        success, buf = cv2.imencode(ext, resized)
        if success:
            buf.tofile(save_path)
            print(f"저장 완료: {save_path} ({new_w}×{new_h})")
        else:
            print(f"[오류] 저장 실패: {save_path}")
