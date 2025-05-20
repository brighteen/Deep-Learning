import cv2
import os

def measure_video_frame_size(video_path=None):
    """
    비디오 파일 또는 웹캠의 프레임 크기(해상도)를 측정하는 함수
    
    Args:
        video_path: 비디오 파일 경로 (None이면 웹캠 사용)
        
    Returns:
        width, height: 프레임 너비와 높이
    """
    
    # 비디오 캡처 객체 생성 (파일 또는 웹캠)
    if video_path is None:
        print("웹캠을 사용합니다.")
        cap = cv2.VideoCapture(0)  # 0은 기본 웹캠
    else:
        # 파일 존재 확인
        if not os.path.exists(video_path):
            print(f"파일이 존재하지 않습니다: {video_path}")
            return None, None
        
        print(f"비디오 파일을 엽니다: {video_path}")
        cap = cv2.VideoCapture(video_path)
    
    # 비디오가 제대로 열렸는지 확인
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        return None, None
    
    # 프레임 크기 가져오기 (방법 1: 속성 사용)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"방법 1 - 프레임 크기: {width} x {height}")
    
    # 프레임 크기 가져오기 (방법 2: 실제 프레임 읽기)
    ret, frame = cap.read()
    if ret:
        h, w, _ = frame.shape
        print(f"방법 2 - 프레임 크기: {w} x {h}")
    
    # 자원 해제
    cap.release()
    
    return width, height

# 사용 예시
if __name__ == "__main__":
    # 비디오 파일 경로 지정 (없으면 None으로 설정하여 웹캠 사용)
    # video_path = "비디오파일경로.mp4"  # 실제 경로로 변경하거나 None으로 설정
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
    
    # 프레임 크기 측정
    width, height = measure_video_frame_size(video_path)
    
    if width is not None and height is not None:
        print(f"최종 프레임 크기: {width} x {height} 픽셀")
        print(f"종횡비: {width/height:.2f}")