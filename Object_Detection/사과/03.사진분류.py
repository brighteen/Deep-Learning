import os
import cv2
import numpy as np

def process_image(image_path,
                  resize_scale=1.0,
                  grabcut_iter=5,
                  hole_area_thresh=100,
                  canny_thresh1=30,
                  canny_thresh2=100,
                  hough_thresh=15,
                  min_line_length=20,
                  max_line_gap=20,
                  line_count_thresh=1,
                  erode_kernel_size=8):
    # 1) 이미지 로드
    with open(image_path, 'rb') as f:
        arr = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")

    # 2) 리사이즈
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*resize_scale), int(h*resize_scale)),
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

    # 4) 마스크 침식으로 핵심부만
    kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    core_mask = cv2.erode(mask2, kernel, iterations=1)
    apple_core = img * core_mask[:, :, None]

    # 5) Canny 엣지
    gray = cv2.cvtColor(apple_core, cv2.COLOR_BGR2GRAY)
    edges_core = cv2.Canny(cv2.GaussianBlur(gray, (5,5),0),
                           canny_thresh1, canny_thresh2)

    # 6) HoughLinesP로 선 검출
    lines = cv2.HoughLinesP(edges_core, 1, np.pi/180, hough_thresh,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    line_count = len(lines) if lines is not None else 0

    # 7) 구멍 검출
    mask_inv = (1 - mask2).astype('uint8')
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_inv, 8)
    hole_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT],
                        stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH],
                        stats[i, cv2.CC_STAT_HEIGHT])
        if area > hole_area_thresh and x>0 and y>0 and x+bw<w and y+bh<h:
            hole_count += 1

    # 8) 이진 분류: 구멍 OR 선 개수 기준
    occluded = (hole_count >= 0) or (line_count >= line_count_thresh)

    # 9) 시각화용 엣지 이미지 생성
    branch_edges = np.zeros_like(edges_core)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            cv2.line(branch_edges, (x1,y1), (x2,y2), 255, 2)

    return occluded, edges_core, branch_edges

if __name__ == "__main__":
    base_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\사과\images"
    for label in ["normal","abnormal"]:
        folder = os.path.join(base_dir, label)
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith((".jpg",".png")): continue
            path = os.path.join(folder, fn)
            occl, edges, branches = process_image(path)
            print(f"[{fn}, {'Yes' if occl else 'No'}]")
            cv2.imshow("Core Edges", edges)
            cv2.imshow("Branch Edges", branches)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
