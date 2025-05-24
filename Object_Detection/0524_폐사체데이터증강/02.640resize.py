import cv2
import os

def resize_frame_from_video(video_path, output_path, frame_number=0):
    """
    영상에서 특정 프레임을 추출하여 640x640으로 리사이징한 후 저장
    """
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("영상을 열 수 없습니다")
        return None
    
    # 전체 프레임 수 확인
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 프레임 번호 확인 및 조정
    if frame_number >= total_frames:
        frame_number = 0
        print(f"요청한 프레임이 총 프레임 수를 초과하여 첫 번째 프레임을 사용합니다.")
    
    # 해당 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("프레임을 읽을 수 없습니다")
        cap.release()
        return None
    
    # 640x640으로 리사이징
    resized_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
    
    # 결과 저장
    cv2.imwrite(output_path, resized_frame)
    
    cap.release()
    print(f"프레임이 {output_path}에 저장되었습니다")
    return output_path

# 경로 설정
video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4"
output_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\resized_frame_640x640.jpg"

# 중앙 부분의 프레임을 가져오기 위해 100번째 프레임 사용 (원하는 프레임으로 변경 가능)
resize_frame_from_video(video_path, output_path, frame_number=100)