import os
from VideoPlayer import VideoPlayer


if __name__ == "__main__":
    """메인 함수"""
    # 직접 경로 지정
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
    model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best.pt"
      # 파일 존재 확인
    if os.path.exists(video_path):
        print(f"영상 파일을 로딩합니다: {os.path.basename(video_path)}")
        
        # 비디오 플레이어 생성 및 실행
        player = VideoPlayer(video_path, model_path, grid_size=3, scale_factor=0.5)
        player.play()
    else:
        print(f"파일이 존재하지 않습니다: {video_path}")