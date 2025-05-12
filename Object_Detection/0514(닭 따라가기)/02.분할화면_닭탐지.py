import cv2
import os
import numpy as np
import time
from ultralytics import YOLO  # YOLOv8 모델을 위한 라이브러리
from PIL import ImageFont, ImageDraw, Image

def put_text_on_image(img, text, position, font_size=30, font_color=(0, 255, 0), font_thickness=2):
    """
    한글을 포함한 텍스트를 이미지에 그립니다.
    
    Args:
        img (numpy.ndarray): 이미지 배열
        text (str): 표시할 텍스트
        position (tuple): 텍스트를 표시할 위치 (x, y)
        font_size (int): 폰트 크기
        font_color (tuple): 폰트 색상 (B, G, R)
        font_thickness (int): 폰트 두께
        
    Returns:
        numpy.ndarray: 텍스트가 추가된 이미지
    """
    # 이미지를 PIL 형식으로 변환
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # 폰트 로드 (Windows에서는 기본 폰트 "malgun.ttf"를 사용)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)  # 윈도우 기본 한글 폰트
    except:
        # 폰트를 찾을 수 없는 경우 기본 폰트 사용
        font = ImageFont.load_default()
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=font_color[::-1])  # RGB -> BGR 변환을 위해 color를 뒤집음
    
    # PIL 이미지를 NumPy 배열로 변환하여 반환
    return np.array(img_pil)

def detect_chickens(model, frame, conf_threshold=0.5):
    """
    YOLO 모델을 사용하여 주어진 프레임에서 닭을 탐지합니다.
    
    Args:
        model: YOLO 모델
        frame: 탐지할 프레임
        conf_threshold: 탐지 확신도 임계값
        
    Returns:
        탐지 결과와 닭의 개수
    """
    # YOLOv8 모델로 탐지
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    
    # 결과에서 닭 개체 수 계산
    boxes = results.boxes
    chicken_count = len(boxes)
    
    # 결과와 닭 개수 반환
    return results, chicken_count

def play_video_with_grid(video_path, model_path, grid_size=5, scale_factor=1.0):
    """
    영상을 그리드로 분할하여 재생하고, YOLO 모델을 사용하여 닭을 탐지합니다.
    
    Args:
        video_path (str): 영상 파일의 경로
        model_path (str): YOLO 모델 파일 경로
        grid_size (int): 분할 그리드 크기 (grid_size x grid_size)
        scale_factor (float): 영상 크기 조절 비율 (기본값: 1.0)
    """
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return
      # YOLOv8 모델 로드
    try:
        model = YOLO(model_path)
        yolo_enabled = True
        print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        yolo_enabled = False
    
    # 탐지 관련 설정
    conf_threshold = 0.5  # 확신도 임계값
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 열기 실패 시
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    # 비디오의 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(1000 / fps)  # 프레임 간 지연시간 (밀리초)
    
    # 비디오 정보 출력
    print(f"원본 영상 크기: {width}x{height}")
    print(f"조절 비율: {scale_factor}")
    print(f"영상 재생 중... (FPS: {fps:.2f})")
    print(f"그리드 크기: {grid_size}x{grid_size}")
    print("조작 방법:")
    print("- 'q' 키: 종료")
    print("- 'g' 키: 그리드 모드/확대 모드 전환")
    print("- '+/-' 키: 확대/축소")
    print("- 'y' 키: YOLO 탐지 켜기/끄기")
    print("- '[/]' 키: 탐지 임계값 조절")
    print("- 스페이스바: 재생/일시정지")
    print("- 'a'/'d' 키: 뒤로/앞으로 5초")
    print("- 마우스 클릭: 그리드 칸 선택/확대")
    
    # 각종 상태 변수
    is_paused = False
    curr_frame_pos = 0
    is_grid_mode = True  # 그리드 모드 여부
    selected_cell = None  # 선택된 그리드 셀 (row, col)
    
    # YOLO 탐지 관련 상태
    yolo_detection_active = False  # 초기에는 YOLO 탐지 비활성화
    
    # 마우스 콜백 함수
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_cell, is_grid_mode
        
        if event == cv2.EVENT_LBUTTONDOWN and is_grid_mode:
            # 그리드 모드에서 마우스 클릭 시 해당 셀 선택
            col = int(x / (width * scale_factor / grid_size))
            row = int(y / (height * scale_factor / grid_size))
            
            # 그리드 범위 내의 셀만 선택 가능
            if 0 <= col < grid_size and 0 <= row < grid_size:
                selected_cell = (row, col)
                is_grid_mode = False
                print(f"선택한 그리드 셀: ({row}, {col})")
    
    # 마우스 콜백 등록
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', mouse_callback)
    
    # 처음 프레임 번호 저장
    curr_frame_pos = 0
    
    while True:
        if not is_paused:
            # 일반 재생 모드: 다음 프레임 읽기
            ret, frame = cap.read()
            curr_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 영상이 끝나면 종료
            if not ret:
                print("영상이 끝났거나 읽기 실패.")
                break
        else:
            # 일시정지 모드: 현재 프레임 유지
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_pos - 1)
            ret, frame = cap.read()
            
            if not ret:
                print("프레임 읽기 실패.")
                break
        
        # 닭 탐지 (YOLO가 활성화된 경우)
        chicken_count = 0
        if yolo_enabled and yolo_detection_active:
            if not is_grid_mode and selected_cell is not None:                # 선택된 셀에서만 닭 탐지
                row, col = selected_cell
                cell_height = height // grid_size
                cell_width = width // grid_size
                x = col * cell_width
                y = row * cell_height
                cell_frame = frame[y:y+cell_height, x:x+cell_width]
                results, chicken_count = detect_chickens(model, cell_frame, conf_threshold)
                  # 탐지 결과를 프레임에 그림 (레이블 크기와 선 두께 조정)
                plotted_cell = results.plot(labels=True, font_size=0.5, line_width=1)  # 레이블 크기와 선 두께 조정
                frame[y:y+cell_height, x:x+cell_width] = plotted_cell
            else:
                # 전체 프레임에서 닭 탐지
                results, chicken_count = detect_chickens(model, frame, conf_threshold)
                
                # 탐지 결과를 프레임에 그림 (레이블 크기와 선 두께 조정)
                frame = results.plot(labels=True, font_size=0.4, line_width=1)  # 레이블 크기와 선 두께 조정
        
        # 화면 표시 준비
        if is_grid_mode:
            # 그리드 모드: 전체 프레임에 그리드 그리기
            display_frame = frame.copy()
            frame_height, frame_width = display_frame.shape[:2]
            
            # 그리드 선 그리기
            cell_height = frame_height // grid_size
            cell_width = frame_width // grid_size
            
            # 가로선 그리기
            for i in range(1, grid_size):
                y = i * cell_height
                cv2.line(display_frame, (0, y), (frame_width, y), (0, 255, 0), 2)
            
            # 세로선 그리기
            for i in range(1, grid_size):
                x = i * cell_width
                cv2.line(display_frame, (x, 0), (x, frame_height), (0, 255, 0), 2)
            
            # 각 셀에 번호 표시
            for row in range(grid_size):
                for col in range(grid_size):
                    x = col * cell_width + cell_width // 2 - 20
                    y = row * cell_height + cell_height // 2
                    cv2.putText(display_frame, f"({row},{col})", (x, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 확대 모드: 선택된 셀만 확대하여 표시
            if selected_cell is not None:
                row, col = selected_cell
                cell_height = height // grid_size
                cell_width = width // grid_size
                x = col * cell_width
                y = row * cell_height
                display_frame = frame[y:y+cell_height, x:x+cell_width].copy()
            else:
                display_frame = frame.copy()
                is_grid_mode = True  # 선택된 셀이 없으면 그리드 모드로 전환
        
        # 크기 조절
        if scale_factor != 1.0:
            display_height, display_width = display_frame.shape[:2]
            new_width = int(display_width * scale_factor)
            new_height = int(display_height * scale_factor)
            display_frame = cv2.resize(display_frame, (new_width, new_height))
        
        # 프레임에 정보 표시
        time_pos = curr_frame_pos / fps  # 현재 시간 위치(초)
        total_time = total_frames / fps   # 총 시간(초)
        
        # 상태 정보 표시 (한글 지원)
        display_frame = put_text_on_image(display_frame, f"배율: {scale_factor:.2f}", (10, 30), 25, (0, 255, 0))
        display_frame = put_text_on_image(display_frame, f"시간: {time_pos:.1f}초 / {total_time:.1f}초", (10, 60), 25, (0, 255, 0))
        display_frame = put_text_on_image(display_frame, f"{'일시정지' if is_paused else '재생 중'}", (10, 90), 25, (0, 255, 0))
        display_frame = put_text_on_image(display_frame, f"모드: {'그리드' if is_grid_mode else '확대'}", (10, 120), 25, (0, 255, 0))
        
        # 선택된 셀 정보 표시
        if not is_grid_mode and selected_cell is not None:
            display_frame = put_text_on_image(display_frame, f"선택 셀: ({selected_cell[0]}, {selected_cell[1]})", 
                        (10, 150), 25, (0, 255, 0))
        
        # YOLO 탐지 정보 표시
        if yolo_enabled:
            color = (0, 255, 255) if yolo_detection_active else (0, 0, 255)
            display_frame = put_text_on_image(display_frame, 
                        f"YOLO: {'켜짐' if yolo_detection_active else '꺼짐'} (임계값: {conf_threshold:.2f})", 
                        (10, 180), 25, color)
            
            if yolo_detection_active:
                display_frame = put_text_on_image(display_frame, f"탐지된 닭: {chicken_count}마리", 
                            (10, 210), 25, (0, 255, 255))
        
        # 프레임 표시
        cv2.imshow('Video', display_frame)
        
        # 키 입력 대기 (일시정지 상태일 때는 더 빠르게 반응하도록)
        wait_time = 1 if is_paused else delay
        key = cv2.waitKey(wait_time) & 0xFF
        
        # 키 입력에 따른 동작
        if key == ord('q'):  # 'q' 키: 종료
            print("사용자가 재생을 중지했습니다.")
            break
        
        elif key == ord('g'):  # 'g' 키: 그리드 모드/확대 모드 전환
            is_grid_mode = not is_grid_mode
            print(f"{'그리드' if is_grid_mode else '확대'} 모드로 전환")
        
        elif key == ord('y'):  # 'y' 키: YOLO 탐지 켜기/끄기
            if yolo_enabled:
                yolo_detection_active = not yolo_detection_active
                print(f"YOLO 탐지: {'활성화' if yolo_detection_active else '비활성화'}")
            else:
                print("YOLO 모델이 로드되지 않았습니다.")
                
        elif key == ord('['):  # '[' 키: 임계값 낮추기
            if yolo_enabled:
                conf_threshold = max(0.1, conf_threshold - 0.1)
                print(f"탐지 임계값 변경: {conf_threshold:.2f}")
                
        elif key == ord(']'):  # ']' 키: 임계값 높이기
            if yolo_enabled:
                conf_threshold = min(0.9, conf_threshold + 0.1)
                print(f"탐지 임계값 변경: {conf_threshold:.2f}")
                
        elif key == ord('+') or key == ord('='):  # '+' 키: 확대
            scale_factor += 0.1
            print(f"확대: {scale_factor:.2f}")
            
        elif key == ord('-'):  # '-' 키: 축소
            if scale_factor > 0.2:  # 너무 작아지지 않도록 제한
                scale_factor -= 0.1
                print(f"축소: {scale_factor:.2f}")
                
        elif key == 32:  # 스페이스바: 재생/일시정지
            is_paused = not is_paused
            print("일시정지" if is_paused else "재생")
            
        elif key == ord('d'):  # d키: 앞으로 이동
            # 5초 앞으로
            target_frame = min(curr_frame_pos + int(fps * 5), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            curr_frame_pos = target_frame
            print(f"앞으로 이동: {target_frame / fps:.1f}초")
            
        elif key == ord('a'):  # a키: 뒤로 이동
            # 5초 뒤로
            target_frame = max(curr_frame_pos - int(fps * 5), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            curr_frame_pos = target_frame
            print(f"뒤로 이동: {target_frame / fps:.1f}초")
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 직접 경로 지정
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4"
    model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt"
    
    # 파일 존재 확인
    if os.path.exists(video_path):
        print(f"영상 파일을 로딩합니다: {os.path.basename(video_path)}")
        
        # 영상 재생 (기본 설정: 0.5배 크기로 축소하여 재생, 5x5 그리드)
        play_video_with_grid(video_path, model_path, grid_size=5, scale_factor=0.5)
    else:
        print(f"파일이 존재하지 않습니다: {video_path}")
