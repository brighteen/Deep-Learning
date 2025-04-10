import os
import cv2
import numpy as np

# -----------------------------
# 1) classify_occlusion 함수 정의
# -----------------------------
def classify_occlusion(image_path,
                       resize_scale=0.5,
                       grabcut_iter=5,
                       hole_area_thresh=200,
                       hough_thresh=20,
                       min_line_length=35,
                       max_line_gap=15,
                       line_count_thresh=1):
    # 이미지 로드 (한글 경로 대응)
    with open(image_path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    # 리사이즈
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*resize_scale), int(h*resize_scale)),
                     interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # GrabCut으로 사과 분할
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, grabcut_iter,
                cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD), 0, 1).astype('uint8')
    apple = img * mask2[:, :, None]

    # 내부 구멍 검출
    mask_inv = (1 - mask2).astype('uint8')
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
    holes = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                       stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area > hole_area_thresh and x>0 and y>0 and x+bw<w and y+bh<h:
            holes += 1

    # 선 검출 (Hough)
    gray = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, hough_thresh,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    line_count = len(lines) if lines is not None else 0

    # 이진 분류
    occluded = (holes > 0) or (line_count >= line_count_thresh)
    return occluded

# -----------------------------
# 2) 전체 이미지 순회 및 결과 출력
# -----------------------------
if __name__ == "__main__":
    base_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\사과\images"
    subdirs = ["normal", "abnormal"]
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for sub in subdirs:
        folder = os.path.join(base_dir, sub)
        if not os.path.isdir(folder):
            continue

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(exts):
                continue
            path = os.path.join(folder, fname)
            try:
                occl = classify_occlusion(path)
                status = "Yes" if occl else "No"
            except Exception as e:
                status = f"Error({e})"
            print(f"[{fname}, {status}]")
