import os
import cv2
import numpy as np

# 설정
base_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\사과\images"
subdirs = ["normal", "abnormal"]

# 처리할 확장자
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for sub in subdirs:
    folder = os.path.join(base_dir, sub)
    if not os.path.isdir(folder):
        print(f"[경고] 폴더가 없습니다: {folder}")
        continue

    for fname in os.listdir(folder):
        name, ext = os.path.splitext(fname)
        # 이미 분할된 파일은 건너뛰기
        if not fname.lower().endswith(EXTS) or name.endswith("_segment"):
            continue

        path = os.path.join(folder, fname)
        # --- 한글 경로 대응 파일 읽기 ---
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[오류] 이미지를 읽을 수 없음: {path}")
            continue

        h, w = img.shape[:2]
        # GrabCut 초기 사각형: 이미지 중앙에 사과가 있다고 가정
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # GrabCut 적용
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # 배경(0,2)은 0, 전경(1,3)은 1로
        mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
        segment = img * mask2[:, :, np.newaxis]

        # 저장
        # save_fname = f"{name}_segment{ext}"
        # save_path = os.path.join(folder, save_fname)
        # success, buf = cv2.imencode(ext, segment)
        # if success:
        #     buf.tofile(save_path)
        #     print(f"분할 저장: {save_path}")
        # else:
        #     print(f"[오류] 분할 저장 실패: {save_path}")
