import cv2

file_path = "Object_Detection/객체탐지 실습/datas/tile_r0_c0.mp4"
cap = cv2.VideoCapture(file_path)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 더 이상 읽을 수 없습니다.")
        break
    
    cv2.imshow("Frame", frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
