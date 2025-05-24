import cv2
import numpy as np
import os

def undistort_video_first_5_seconds(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("영상을 열 수 없습니다")
        return
    
    # 영상 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 5초에 해당하는 프레임 수 계산
    frames_to_process = int(fps * 5)
    print(f"처리할 프레임 수: {frames_to_process} (FPS: {fps})")
    
    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = os.path.join(output_path, "첫5초_왜곡보정.mp4")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # 카메라 매트릭스와 왜곡 계수
    camera_matrix = np.array([
        [width*0.8, 0, width/2],
        [0, height*0.8, height/2],
        [0, 0, 1]
    ])
    
    # 왜곡 계수 (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([-0.3, 0.1, 0, 0, -0.02])
    
    # 새 카메라 매트릭스
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 0, (width, height))
    
    # 매핑 테이블 계산
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, newcameramtx, (width, height), 5)
    
    frame_count = 0
    
    while frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 왜곡 보정 적용
        undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # 결과 저장
        out.write(undistorted)
        
        frame_count += 1
        if frame_count % 30 == 0:  # 30프레임마다 상태 출력
            print(f"처리 중: {frame_count}/{frames_to_process} 프레임")
    
    cap.release()
    out.release()
    print(f"처리 완료: 첫 5초({frame_count} 프레임)가 {output_file}에 저장되었습니다")

# 실행
input_video = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4"
output_dir = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\왜곡보정"

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

undistort_video_first_5_seconds(input_video, output_dir)