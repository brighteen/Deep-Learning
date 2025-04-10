import os
import cv2
import numpy as np

def process_image(image_path,
                  resize_scale=1.0,
                  grabcut_iter=5,
                  hole_area_thresh=100,
                  canny_thresh1=30,
                  canny_thresh2=100,
                  hough_thresh=30,
                  min_line_length=30,
                  max_line_gap=20,
                  line_count_thresh=2,
                  erode_kernel_size=21):
    """
    1) GrabCut으로 사과 분할
    2) 마스크를 침식(erode)하여 사과 핵심부(core)만 추출
    3) 핵심부에서 Canny 엣지 검출
    4) 핵심부 엣지에서 HoughLinesP로 가지로 간주되는 선 검출
    5) hole_count>0 AND line_count>=threshold 이면 occluded
    Returns:
      occluded: bool
      edges_core: 사과 핵심부 전체 엣지
      branch_edges: 가지로 인식된 엣지
    """
    # 1) 이미지 로드 (한글 경로 대응)
    with open(image_path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")

    # 2) 리사이즈
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * resize_scale), int(h * resize_scale)),
                     interpolation=cv2.INTER_AREA)

    # 3) GrabCut으로 사과 분할
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, grabcut_iter,
                cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD), 0, 1).astype('uint8')

    # 4) 마스크 침식으로 핵심부만 남기기
    kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    core_mask = cv2.erode(mask2, kernel, iterations=1)
    apple_core = img * core_mask[:, :, None]

    # 5) 핵심부에서 Canny 엣지 검출
    gray_core = cv2.cvtColor(apple_core, cv2.COLOR_BGR2GRAY)
    blurred_core = cv2.GaussianBlur(gray_core, (5,5), 0)
    edges_core = cv2.Canny(blurred_core, canny_thresh1, canny_thresh2)

    # 6) HoughLinesP로 가지 엣지 검출
    lines = cv2.HoughLinesP(
        edges_core, 1, np.pi/180, hough_thresh,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    branch_edges = np.zeros_like(edges_core)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(branch_edges, (x1, y1), (x2, y2), 255, 2)
    line_count = len(lines) if lines is not None else 0

    # 7) hole_count 계산 (구멍 검출)
    mask_inv = (1 - mask2).astype('uint8')
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
    hole_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT],
                        stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH],
                        stats[i, cv2.CC_STAT_HEIGHT])
        if area > hole_area_thresh and x>0 and y>0 and x+bw<w and y+bh<h:
            hole_count += 1

    # 8) 최종 이진 분류: 구멍 AND 선 개수 기준
    occluded = (hole_count > 0) and (line_count >= line_count_thresh)

    return occluded, edges_core, branch_edges

if __name__ == "__main__":
    base_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\사과\images"
    subdirs = ["normal", "abnormal"]
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for label in subdirs:
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(exts):
                continue
            path = os.path.join(folder, fname)
            try:
                occl, edges_core, branch_edges = process_image(path)
                status = "Yes" if occl else "No"
            except Exception as e:
                status = f"Error({e})"
                edges_core = np.zeros((100,100), dtype=np.uint8)
                branch_edges = edges_core.copy()

            print(f"[{fname}, {status}]")

            # 시각화
            cv2.imshow("Apple Core Edges", edges_core)
            cv2.imshow("Branch Edges", branch_edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
