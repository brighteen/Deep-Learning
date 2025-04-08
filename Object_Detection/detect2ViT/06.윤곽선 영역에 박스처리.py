# 6. 윤곽선 영역에 박스처리하기(분류 무시)
import cv2
file_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\detect2ViT\datas\tile_r0_c3.mp4"

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
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 조건에 맞는 윤곽선 영역에 바운딩 박스 그리기 (임시로 조건 완화)
        if area > 50:  # 일단 50보다 큰 모든 영역 표시해보기
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Bounding Boxes", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
        
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
