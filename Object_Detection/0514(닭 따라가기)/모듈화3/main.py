import os
from VideoPlayer import VideoPlayer

if __name__ == "__main__":
    """메인 함수"""
    # 직접 경로 지정 (기존 모듈화2와 동일한 경로 사용)
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
    model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt"
    
    # 파일 존재 확인
    if os.path.exists(video_path):
        print(f"영상 파일을 로딩합니다: {os.path.basename(video_path)}")
        
        # 비디오 플레이어 설정값
        scale_factor = 0.4  # 화면 크기 비율
        detection_interval = 30  # 탐지 간격 (프레임)
        
        # 비디오 플레이어 생성
        player = VideoPlayer(video_path, model_path, scale_factor=scale_factor, detection_interval=detection_interval)
        
        # 비디오 초기화
        player.initialize_video()
        
        # 영상 분석 시간 범위 설정 (초 단위)
        player.set_video_range(start_time=10, end_time=40)  # 10초~40초 구간만 분석
        
        # 프레임 ROI(관심 영역) 설정 - 화면의 일부 영역만 처리
        # x1, y1: 시작 좌표, x2, y2: 종료 좌표
        # None으로 설정하면 전체 영상 크기 사용
        player.set_frame_roi(x1=100, y1=100, x2=600, y2=600)
        
        # 재생 속도 설정 (1.0이 기본 속도, 2.0이면 2배 빠르게)
        player.set_playback_speed(speed_factor=1.0)
        
        # 비디오 재생 시작
        print("ESC 키를 누르면 종료합니다. 'a'/'d' 키로 5초 앞/뒤로 이동할 수 있습니다.")
        player.play()
    else:
        print(f"파일이 존재하지 않습니다: {video_path}")