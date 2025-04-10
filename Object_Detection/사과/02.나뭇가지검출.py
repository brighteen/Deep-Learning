import cv2
import numpy as np

def classify_occlusion(image_path,
                       resize_scale=0.5,
                       grabcut_iter=5,
                       hole_area_thresh=50,   # 구멍 최소 픽셀 수
                       hough_thresh=20,
                       min_line_length=30,
                       max_line_gap=10,
                       line_count_thresh=1):
    """
    Args:
      image_path: 이미지 파일 경로
      resize_scale: 처리 속도를 위해 리사이즈 비율
      grabcut_iter: GrabCut 반복 횟수
      hole_area_thresh: 내부 구멍 검출 시 최소 픽셀 수
      line_count_thresh: 선 검출 시 이 기준 이상이면 'occluded'
    Returns:
      occluded: bool (True면 물체 있음)
      debug: dict (중간 결과)
    """
    # 1) 이미지 로드 (한글 경로 안전하게)
    with open(image_path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    # 2) 리사이즈
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*resize_scale), int(h*resize_scale)),
                     interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # 3) GrabCut으로 사과 분할
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, grabcut_iter,
                cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==cv2.GC_BGD)|(mask==cv2.GC_PR_BGD), 0, 1).astype('uint8')
    apple = img * mask2[:, :, None]

    # 4) 내부 구멍(hole) 검출
    #    사과 마스크(mask2)의 내부 배경 픽셀(0)을 찾고,
    #    경계에 닿지 않는 컴포넌트를 구멍으로 간주
    mask_inv = (1 - mask2).astype('uint8')
    # 레이블링
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
    holes = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                       stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # 경계에 닿지 않고, 충분히 큰 면적이면 구멍
        if area > hole_area_thresh and x>0 and y>0 and x+bw<w and y+bh<h:
            holes += 1

    # 5) 선(line) 검출 (Hough)
    gray = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_thresh,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    line_count = len(lines) if lines is not None else 0

    # 6) 최종 판단
    occluded = (holes > 0) or (line_count >= line_count_thresh)

    debug = {
        'resized_shape': apple.shape[:2],
        'hole_count': holes,
        'line_count': line_count,
        'mask2': mask2,         # 사과 분할 마스크
        'apple_seg': apple,     # 분할된 사과 이미지
        'edges': edges
    }
    return occluded, debug

if __name__ == "__main__":
    # img_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\사과\images\normal\n05_resized.jpg"
    img_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\사과\images\abnormal\abn01_resized.jpg"
    occl, dbg = classify_occlusion(img_path)
    print("Occluded by object?" , "Yes" if occl else "No")
    print("Hole count:", dbg['hole_count'])
    print("Line count:", dbg['line_count'])
    # 디버그 이미지 보기 (예시)
    cv2.imshow("Apple Seg", dbg['apple_seg'])
    cv2.imshow("Mask", dbg['mask2']*255)
    cv2.imshow("Edges", dbg['edges'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
