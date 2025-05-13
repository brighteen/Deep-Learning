import os
import cv2
import numpy as np
import time
import argparse
import pandas as pd
from datetime import datetime

# 모듈 가져오기
from utils import put_text_on_image, create_grid_display, get_cell_frame
from chicken_detector import load_yolo_model, detect_chickens, track_chickens
from chicken_tracker import ChickenTracker
from visualization import add_status_info, add_tracking_info, show_skipped_frame_info, create_prompt_frame

def play_video_with_tracking(video_path, model_path, grid_size=5, scale_factor=1.0, max_time=60, frame_skip=5, target_cell=(2,1)):
    """
    영상을 그리드로 분할하여 재생하고, YOLO 모델을 사용하여 닭을 추적합니다.
    특정 그리드 셀(기본값: 2,1)에 집중하여 닭 ID 추적을 수행합니다.
    
    Args:
        video_path (str): 영상 파일의 경로
        model_path (str): YOLO 모델 파일 경로
        grid_size (int): 분할 그리드 크기 (grid_size x grid_size)
        scale_factor (float): 영상 크기 조절 비율
        max_time (float): 최대 재생 시간(초)
        frame_skip (int): 탐지/추적을 수행할 프레임 간격 (높을수록 빠르지만 정확도 감소)
        target_cell (tuple): 추적할 대상 셀 좌표 (row, col), 기본값은 (2,1)
    """
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return
    
    # YOLOv8 모델 로드
    model, yolo_enabled = load_yolo_model(model_path)
    if yolo_enabled:
        print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
    
    # 닭 추적을 위한 ChickenTracker 인스턴스 생성
    tracker = ChickenTracker(max_disappeared=10, iou_threshold=0.3)
    
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
    print(f"최대 재생 시간: {max_time}초")
    print(f"프레임 스킵: {frame_skip} (탐지는 {frame_skip}프레임마다 수행)")
    print("조작 방법:")
    print("- 'q' 키: 종료")
    print("- 'g' 키: 그리드 모드/확대 모드 전환")
    print("- '+/-' 키: 확대/축소")
    print("- 'y' 키: YOLO 탐지 켜기/끄기")
    print("- '[/]' 키: 탐지 임계값 조절")
    print("- 스페이스바: 재생/일시정지")
    print("- 'a'/'d' 키: 뒤로/앞으로 5초")
    print("- 'f' 키: 프레임 스킵 값 변경")
    
    # 각종 상태 변수
    is_paused = False
    curr_frame_pos = 0
    is_grid_mode = False  # 확대 모드로 시작
    selected_cell = target_cell  # 지정된 셀 선택
    
    # YOLO 탐지 관련 상태
    yolo_detection_active = True  # 초기에 YOLO 탐지 활성화
    
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
                # 클릭한 셀이 target_cell과 다른 경우 경고 표시
                if selected_cell != target_cell:
                    print(f"경고: 추적 대상 셀은 {target_cell}이지만, 선택한 셀은 ({row}, {col})입니다.")
                print(f"선택한 그리드 셀: ({row}, {col})")
    
    # 마우스 콜백 등록
    cv2.namedWindow('Video with Tracking')
    cv2.setMouseCallback('Video with Tracking', mouse_callback)
    
    # 처음 프레임 번호 저장
    curr_frame_pos = 0
    frame_count = 0  # 프레임 카운터 추가 (스킵에 사용)
    
    while True:
        # 최대 재생 시간 제한
        if curr_frame_pos / fps > max_time:
            print(f"설정된 최대 재생 시간({max_time}초)에 도달했습니다.")
            break
        
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
                
        # 닭 탐지 및 추적 (YOLO가 활성화된 경우)
        chicken_count = 0
        chicken_ids = []
        chicken_sets = []
        should_detect = frame_count % frame_skip == 0  # frame_skip 간격으로 탐지 수행
        frame_count += 1
        
        time_pos = curr_frame_pos / fps  # 현재 시간 위치(초)
        
        if yolo_enabled and yolo_detection_active:
            if not is_grid_mode and selected_cell == target_cell:  # 지정된 셀에서만 닭 추적
                # 셀 영역 추출
                y, x, cell_height, cell_width, cell_frame = get_cell_frame(frame, selected_cell, grid_size)
                
                if should_detect:  # frame_skip 간격으로만 탐지/추적 수행
                    # 닭 추적 수행
                    results, chicken_count = track_chickens(model, cell_frame, conf_threshold)
                    
                    # 추적 결과 처리
                    if results.boxes.id is not None:
                        # ID와 박스 정보 추출
                        ids = results.boxes.id.int().cpu().tolist()
                        boxes = results.boxes.xyxy.cpu().tolist()
                        
                        # ChickenTracker 업데이트
                        current_time = curr_frame_pos / fps
                        chicken_ids, chicken_sets = tracker.update(curr_frame_pos, current_time, boxes, ids)
                    
                    # 결과 시각화
                    plotted_cell = tracker.visualize_tracking(cell_frame, results)
                    
                    # 결과를 클래스에 저장 (다음 프레임에서 사용하기 위해)
                    tracker.last_results = plotted_cell.copy()
                    tracker.last_time = time_pos
                    
                    # 결과를 프레임에 적용
                    frame[y:y+cell_height, x:x+cell_width] = plotted_cell
                else:
                    # 탐지를 건너뛰는 프레임에서는 이전 결과 사용
                    frame = show_skipped_frame_info(frame, (y, x, cell_height, cell_width), tracker, time_pos)
            
            elif is_grid_mode and should_detect:  # 그리드 모드이고 탐지 프레임에서만
                # 전체 프레임에서 닭 탐지 (추적 없음)
                results, chicken_count = detect_chickens(model, frame, conf_threshold)
                
                # 탐지 결과를 프레임에 그림 (레이블 크기와 선 두께 조정)
                frame = results.plot(labels=True, font_size=0.4, line_width=1)
        
        # 화면 표시 준비
        if is_grid_mode:
            # 그리드 모드: 전체 프레임에 그리드 그리기
            display_frame = create_grid_display(frame, grid_size, target_cell)
        else:
            # 확대 모드: 선택된 셀만 확대하여 표시
            if selected_cell is not None:
                y, x, cell_height, cell_width, display_frame = get_cell_frame(frame, selected_cell, grid_size)
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
        total_time = total_frames / fps   # 총 시간(초)
        
        # 상태 정보 준비
        info = {
            "배율": scale_factor,
            "시간": f"{time_pos:.1f}초 / {total_time:.1f}초",
            "상태": "일시정지" if is_paused else "재생 중",
            "모드": "그리드" if is_grid_mode else "확대",
            "프레임 스킵": frame_skip
        }
        
        # 상태 정보 표시
        display_frame = add_status_info(display_frame, info)
        
        # 선택된 셀 정보 표시
        if not is_grid_mode and selected_cell is not None:
            display_frame = put_text_on_image(display_frame, 
                          f"선택 셀: ({selected_cell[0]}, {selected_cell[1]})", 
                          (10, 150), 25, (0, 255, 0))
        
        # YOLO 탐지 정보 표시
        if yolo_enabled:
            color = (0, 255, 255) if yolo_detection_active else (0, 0, 255)
            display_frame = put_text_on_image(display_frame, 
                          f"YOLO: {'켜짐' if yolo_detection_active else '꺼짐'} (임계값: {conf_threshold:.2f})", 
                          (10, 180), 25, color)
            
            if yolo_detection_active:
                display_frame = put_text_on_image(display_frame, 
                              f"탐지된 닭: {chicken_count}마리", 
                              (10, 210), 25, (0, 255, 255))
                
                # 추적 정보 표시
                if selected_cell == target_cell and not is_grid_mode:
                    display_frame = add_tracking_info(display_frame, tracker)
        
        # 프레임 표시
        cv2.imshow('Video with Tracking', display_frame)
        
        # 키 입력 대기 (일시정지 상태일 때는 더 빠르게 반응하도록)
        wait_time = 1 if is_paused else delay
        key = cv2.waitKey(wait_time) & 0xFF
        
        # 키 입력에 따른 동작
        if key == ord('q'):  # 'q' 키: 종료
            print("사용자가 재생을 중지했습니다.")
            
            # 추적 데이터가 있는 경우 ID 집합 정보 출력
            if tracker.history:
                # ID 집합 정보 출력 (CSV 저장은 프로그램 종료 시점에 수행됨)
                tracker.print_stats()
            else:
                print("기록된 추적 데이터가 없습니다.")
            
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
                
        elif key == ord('f'):  # 'f' 키: 프레임 스킵 값 변경
            # 화면에 프롬프트 표시
            prompt_message = f"현재 프레임 스킵: {frame_skip}, + 키로 증가, - 키로 감소"
            prompt_frame = create_prompt_frame(display_frame, prompt_message)
            cv2.imshow('Video with Tracking', prompt_frame)
            
            # 키 입력 대기
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('+') or key == ord('='):
                frame_skip = min(15, frame_skip + 1)
                print(f"프레임 스킵을 {frame_skip}으로 증가했습니다.")
            elif key == ord('-'):
                frame_skip = max(1, frame_skip - 1)
                print(f"프레임 스킵을 {frame_skip}으로 감소했습니다.")
            else:
                print("프레임 스킵 변경이 취소되었습니다.")
                
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
      # 종료 시 닭 ID 변화 정보 출력 및 저장
    print("\n프로그램 종료. 닭 ID 변화 정보를 출력합니다...")
    if len(tracker.history) > 0:
        # 닭 ID 변화 정보 출력
        tracker.print_chicken_id_changes()
        
        # 닭 ID 변화 정보 CSV 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"chicken_tracking_{timestamp}.csv"
        details_csv_path = f"chicken_tracking_{timestamp}_details.csv"
        
        # 기본 CSV 파일 저장
        saved_file = tracker.export_to_csv(csv_path, fps)
        if saved_file:
            print(f"닭 ID 변화 정보를 {saved_file}에 저장했습니다.")
        
        # 각 닭의 ID 세트 정보를 별도 CSV로 저장
        try:
            chicken_ids_data = []
            for i, id_set in enumerate(tracker.id_sets):
                sorted_ids = sorted(list(id_set))
                chicken_ids_data.append({
                    'chicken_number': i+1,
                    'id_set': str(sorted_ids),
                    'count': len(id_set)
                })
            
            df_details = pd.DataFrame(chicken_ids_data)
            df_details.to_csv(details_csv_path, index=False)
            print(f"닭 ID 세부 정보를 {details_csv_path}에 저장했습니다.")
        except Exception as e:
            print(f"세부 정보 CSV 파일 저장 중 오류 발생: {e}")
    else:
        print("기록된 추적 데이터가 없습니다. CSV 파일을 저장하지 않았습니다.")
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 명령행 인자 파서 생성
    parser = argparse.ArgumentParser(description="닭 ID 추적 프로그램")
    parser.add_argument("--video", default=r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4", 
                        help="영상 파일 경로")
    parser.add_argument("--model", default=r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt", 
                        help="YOLO 모델 경로")
    parser.add_argument("--grid_size", type=int, default=5, help="그리드 크기 (NxN)")
    parser.add_argument("--scale", type=float, default=0.7, help="영상 크기 조절 비율")
    parser.add_argument("--max_time", type=int, default=60, help="최대 재생 시간(초)")
    parser.add_argument("--frame_skip", type=int, default=30, help="프레임 건너뛰기 (높을수록 빠름)")
    parser.add_argument("--cell", default="2,1", help="추적할 그리드 셀 좌표 (row,col)")
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 셀 좌표 파싱
    try:
        row, col = map(int, args.cell.split(','))
        target_cell = (row, col)
    except:
        print("셀 좌표 형식이 잘못되었습니다. 기본값 (2,1)을 사용합니다.")
        target_cell = (2, 1)
    
    # 파일 존재 확인
    video_path = args.video
    model_path = args.model
    
    if os.path.exists(video_path):
        print(f"영상 파일을 로딩합니다: {os.path.basename(video_path)}")
        print(f"그리드 크기: {args.grid_size}x{args.grid_size}, 추적 셀: {target_cell}")
        print(f"프레임 스킵: {args.frame_skip}, 최대 재생 시간: {args.max_time}초")
        
        # 영상 재생 및 닭 추적
        play_video_with_tracking(
            video_path, 
            model_path, 
            grid_size=args.grid_size, 
            scale_factor=args.scale, 
            max_time=args.max_time, 
            frame_skip=args.frame_skip,
            target_cell=target_cell
        )
    else:
        print(f"파일이 존재하지 않습니다: {video_path}")
