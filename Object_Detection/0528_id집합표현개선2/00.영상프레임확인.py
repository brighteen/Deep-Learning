import cv2

# 영상 경로
video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"

# 비디오 객체 생성
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
else:
    # 총 프레임 수 확인
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # FPS 확인
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 영상 길이 계산 (초)
    duration_seconds = total_frames / fps
    
    # 시간 형식으로 변환
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    
    print(f"총 프레임 수: {total_frames}")
    print(f"FPS: {fps}")
    print(f"영상 길이: {minutes}분 {seconds}초 ({duration_seconds:.2f}초)")

# 리소스 해제
cap.release()