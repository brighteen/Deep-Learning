import os
import cv2
import numpy as np

def process_image(image_path,
                  resize_scale=1.0,
                  grabcut_iter=5,
                  canny_thresh1=30,
                  canny_thresh2=100,
                  hough_thresh=15,
                  min_line_length=20,
                  max_line_gap=20,
                  line_count_thresh=1):
    """
    1) GrabCut으로 사과 분할 → mask2
    2) mask2에서 가장 큰 컨투어로 최소 외접 원 구하기
    3) Canny 엣지 검출 → edges_core
    4) 원 경계(두께 ring_width) 제거 → edges_no_circle
    5) edges_no_circle에서 HoughLinesP로 선 검출 → branch_edges
    6) 선 개수 기준으로 이진 분류
    """
    # 1) 이미지 로드 (한글 경로 안전)
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

    # 4) 가장 큰 컨투어의 최소 외접 원 계산
    binary = (mask2*255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No apple contour found")
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    center = (int(cx), int(cy))
    radius = int(radius)

    # 5) Canny 엣지 검출 (사과 내부)
    apple = img * mask2[:, :, None]
    gray = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges_core = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

    # 6) 원 경계만큼 마스크(두께 ring_width) 생성 후 제거
    ring_width = 5
    circle_mask = np.zeros_like(edges_core)
    cv2.circle(circle_mask, center, radius, 255, thickness=ring_width)
    # 조금 확장
    circle_mask = cv2.dilate(circle_mask, np.ones((3,3),np.uint8), iterations=1)
    edges_no_circle = cv2.bitwise_and(edges_core, cv2.bitwise_not(circle_mask))

    # 7) HoughLinesP로 가지 엣지 검출
    lines = cv2.HoughLinesP(
        edges_no_circle, 1, np.pi/180, hough_thresh,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    branch_edges = np.zeros_like(edges_core)
    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for x1,y1,x2,y2 in lines[:,0]:
            cv2.line(branch_edges, (x1,y1), (x2,y2), 255, 2)

    # 8) 이진 분류: 선 개수 기준
    occluded = (line_count >= line_count_thresh)
    return occluded, edges_core, edges_no_circle, branch_edges

if __name__ == "__main__":
    base_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\사과\images"
    for label in ["normal","abnormal"]:
        folder = os.path.join(base_dir, label)
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith((".jpg",".png")): continue
            path = os.path.join(folder, fn)
            try:
                occl, edges_core, edges_no_circle, branch_edges = process_image(path)
                status = "Yes" if occl else "No"
            except Exception as e:
                status = f"Error({e})"
                edges_core = edges_no_circle = branch_edges = np.zeros((100,100),np.uint8)

            print(f"[{fn}, {status}]")
            # 시각화
            cv2.imshow("All Edges", edges_core)
            cv2.imshow("No Circle Edges", edges_no_circle)
            cv2.imshow("Branch Edges", branch_edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
