import cv2
file_path = "Object_Detection/객체탐지 실습/datas/tile_r0_c2.mp4"
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
    
    cv2.imshow("Frame Difference", frame_diff)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Original", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
