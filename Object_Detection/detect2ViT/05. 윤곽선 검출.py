# 윤곽선(Contour) 검출 및 면적 출력 확인
import cv2
file_path = "Object_Detection/객체탐지 실습/datas/tile_r0_c3.mp4"

cap = cv2.VideoCapture(file_path)
ret, prev_frame = cap.read()
if not ret:
    print("첫 번째 프레임 읽기 실패")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"검출된 윤곽선 개수: {len(contours)}")
    
    # 각 윤곽선의 면적 출력
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print("면적:", area)
    
    cv2.imshow("Threshold", thresh)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
        
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
