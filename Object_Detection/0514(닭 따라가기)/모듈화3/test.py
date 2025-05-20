import cv2
import os
from ultralytics import YOLO

def main():
    """영상을 불러와 YOLO 모델로 객체 탐지를 수행하는 간단한 예제"""
    # 파일 경로 설정
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
    model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt"
    
    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return
    
    # YOLO 모델 로드
    try:
        model = YOLO(model_path)
        print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        return
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 열기 실패 시
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보 출력
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"영상 크기: {width}x{height}, FPS: {fps:.2f}")
      # 화면 창 생성
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    # 창 크기를 작게 조정 (원본 크기의 50%로 변경)
    cv2.resizeWindow("Object Detection", int(width * 0.3), int(height * 0.3))
    
    # 탐지 설정
    conf_threshold = 0.5  # 탐지 확신도 임계값
    
    while True:        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("비디오의 끝에 도달했습니다.")
            break
        
        # 관심 영역 설정 (필요한 부분만 잘라서 사용)
        # frame = frame[y시작:y끝, x시작:x끝] 형식으로 사용
        frame = frame[0:1000, 200:1000]  # 프레임 상단 왼쪽 400x400 영역만 사용
        
        # 객체 탐지 수행
        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
          # 탐지 결과를 프레임에 그리기
        detected_frame = results.plot(labels=True, font_size=0.6, line_width=2)
        
        # 필요에 따라 디스플레이용 프레임 크기 조정 (더 크게 보이게)
        # 400x400 크기 영역을 600x600으로 확대 (선택적)
        # detected_frame = cv2.resize(detected_frame, (600, 600))
        
        # 탐지된 객체 수 계산
        chicken_count = len(results.boxes)
        
        # 화면에 객체 수 표시
        cv2.putText(detected_frame, f"Chickens: {chicken_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 화면에 표시
        cv2.imshow("Object Detection", detected_frame)
        
        # 키 입력 처리 (q: 종료, 스페이스바: 일시정지)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("사용자가 종료했습니다.")
            break
        elif key == ord(' '):
            print("일시정지됨. 계속하려면 아무 키나 누르세요.")
            cv2.waitKey(0)
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()