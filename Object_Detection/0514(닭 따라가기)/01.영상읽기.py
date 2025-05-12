import cv2
import os
import numpy as np
import time
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

def play_video(video_path, scale_factor=1.0, crop_region=None):
    """
    영상을 불러와 재생합니다.
    
    Args:
        video_path (str): 영상 파일의 경로
        scale_factor (float): 영상 크기 조절 비율 (기본값: 1.0)
        crop_region (tuple): 자를 영역 (x, y, width, height) 형식의 튜플, None이면 전체 프레임 표시
        
    조작 방법:
        - 'q' 키: 종료
        - 'c' 키: 전체/부분 전환
        - 's' 키: 영역 선택 모드 활성화 (마우스로 드래그하여 원하는 영역 선택)
        - '+/-' 키: 확대/축소
        - 스페이스바: 재생/일시정지
        - 'a'/'d' 키: 뒤로/앞으로 5초
        - 마우스 드래그: 화면 이동
    """
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return
    
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
    
    # 자르기 영역이 지정되지 않은 경우 전체 프레임 사용
    if crop_region is None:
        crop_region = (0, 0, width, height)
    
    # 비디오 정보 출력
    print(f"원본 영상 크기: {width}x{height}")
    print(f"선택한 영역: {crop_region}")
    print(f"조절 비율: {scale_factor}")
    print(f"영상 재생 중... (FPS: {fps:.2f})")
    print("조작 방법:")
    print("- 'q' 키: 종료")
    print("- 'c' 키: 전체/부분 전환 (기본 중앙 영역)")
    print("- 's' 키: 영역 선택 모드 활성화 (마우스로 드래그)")
    print("- '+/-' 키: 확대/축소")
    print("- 스페이스바: 재생/일시정지")
    print("- 'a'/'d' 키: 뒤로/앞으로 5초")
    print("- 마우스 드래그: 화면 이동")
    
    # 마우스 이벤트를 처리하기 위한 변수들
    mouse_data = {
        'is_dragging': False,
        'drag_start_x': 0,
        'drag_start_y': 0,
        'crop_region': crop_region,
        'selection_mode': False,     # 영역 선택 모드
        'selection_start_x': 0,      # 선택 영역의 시작점 x
        'selection_start_y': 0,      # 선택 영역의 시작점 y
        'selection_end_x': 0,        # 선택 영역의 끝점 x
        'selection_end_y': 0,        # 선택 영역의 끝점 y
        'selecting': False           # 선택 중인지 여부
    }
    
    # 전체 프레임 표시 모드인지 여부 (True면 전체, False면 자른 영역)
    show_full_frame = crop_region == (0, 0, width, height)
    
    # 재생 관련 상태 변수
    is_paused = False
    curr_frame_pos = 0
    
    # 마우스 콜백 함수
    def mouse_callback(event, x, y, flags, param):
        nonlocal scale_factor, show_full_frame
        
        if mouse_data['selection_mode']:
            # 영역 선택 모드일 때
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_data['selecting'] = True
                mouse_data['selection_start_x'] = x
                mouse_data['selection_start_y'] = y
                mouse_data['selection_end_x'] = x
                mouse_data['selection_end_y'] = y
                
            elif event == cv2.EVENT_MOUSEMOVE and mouse_data['selecting']:
                mouse_data['selection_end_x'] = x
                mouse_data['selection_end_y'] = y
                
            elif event == cv2.EVENT_LBUTTONUP:
                mouse_data['selecting'] = False
                
                # 선택 영역 계산
                start_x = min(mouse_data['selection_start_x'], mouse_data['selection_end_x'])
                start_y = min(mouse_data['selection_start_y'], mouse_data['selection_end_y'])
                end_x = max(mouse_data['selection_start_x'], mouse_data['selection_end_x'])
                end_y = max(mouse_data['selection_start_y'], mouse_data['selection_end_y'])
                
                select_width = end_x - start_x
                select_height = end_y - start_y
                
                # 영역 크기가 충분히 큰지 확인
                if select_width > 10 and select_height > 10:
                    # 현재 화면 크기에서 원본 비디오 크기로 좌표 변환
                    if show_full_frame:
                        # 전체화면 모드에서 선택한 경우
                        orig_x = int(start_x / scale_factor)
                        orig_y = int(start_y / scale_factor)
                        orig_width = int(select_width / scale_factor)
                        orig_height = int(select_height / scale_factor)
                    else:
                        # 이미 자른 영역 내에서 선택한 경우
                        curr_x, curr_y, _, _ = mouse_data['crop_region']
                        orig_x = curr_x + int(start_x / scale_factor)
                        orig_y = curr_y + int(start_y / scale_factor)
                        orig_width = int(select_width / scale_factor)
                        orig_height = int(select_height / scale_factor)
                    
                    # 선택 영역이 비디오 프레임 내에 있도록 제한
                    orig_x = min(max(0, orig_x), width)
                    orig_y = min(max(0, orig_y), height)
                    orig_width = min(orig_width, width - orig_x)
                    orig_height = min(orig_height, height - orig_y)
                    
                    # 유효한 영역 설정
                    mouse_data['crop_region'] = (orig_x, orig_y, orig_width, orig_height)
                    show_full_frame = False
                    mouse_data['selection_mode'] = False
                    print(f"선택한 영역으로 변경: {mouse_data['crop_region']}")
                else:
                    mouse_data['selection_mode'] = False
                    print("영역이 너무 작습니다. 다시 시도하세요.")
        else:
            # 일반 드래그 모드 (화면 이동)
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_data['is_dragging'] = True
                mouse_data['drag_start_x'] = x
                mouse_data['drag_start_y'] = y
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if mouse_data['is_dragging'] and not show_full_frame:
                    # 드래그 중이고 부분 영역 모드일 때만 이동
                    dx = mouse_data['drag_start_x'] - x
                    dy = mouse_data['drag_start_y'] - y
                    
                    # 원본 좌표
                    orig_x, orig_y, w, h = mouse_data['crop_region']
                    
                    # 새 좌표 계산 (경계 체크)
                    new_x = min(max(0, orig_x + int(dx/scale_factor)), width - w)
                    new_y = min(max(0, orig_y + int(dy/scale_factor)), height - h)
                    
                    # 업데이트된 좌표
                    mouse_data['crop_region'] = (new_x, new_y, w, h)
                    
                    # 드래그 시작점 업데이트
                    mouse_data['drag_start_x'] = x
                    mouse_data['drag_start_y'] = y
                    
            elif event == cv2.EVENT_LBUTTONUP:
                mouse_data['is_dragging'] = False
    
    # 마우스 콜백 등록
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', mouse_callback)
    
    # 처음 프레임 번호 저장
    curr_frame_pos = 0
    
    while True:
        # 현재 crop_region 업데이트
        crop_region = mouse_data['crop_region']
        
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
        
        # 프레임 처리
        if show_full_frame:
            # 전체 프레임 표시 (크기 조절만)
            display_frame = frame.copy()
        else:
            # 자른 영역 표시
            x, y, w, h = crop_region
            # 영역이 프레임 범위를 벗어나지 않도록 확인
            x = min(max(0, x), width - 1)
            y = min(max(0, y), height - 1)
            w = min(w, width - x)
            h = min(h, height - y)
            
            # 잘린 영역이 유효한지 확인
            if w > 0 and h > 0:
                display_frame = frame[y:y+h, x:x+w].copy()
            else:
                display_frame = frame.copy()
                show_full_frame = True
        
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
        
        # 부분 영역을 보고 있는 경우 표시
        if not show_full_frame:
            display_frame = put_text_on_image(display_frame, f"영역: ({crop_region[0]}, {crop_region[1]}, {crop_region[2]}, {crop_region[3]})", 
                        (10, 120), 25, (0, 255, 0))
        
        # 선택 모드 상태 표시
        if mouse_data['selection_mode']:
            display_frame = put_text_on_image(display_frame, "영역 선택 모드: 마우스로 드래그하세요", 
                        (10, 150), 25, (0, 0, 255))
            
            # 선택 중인 경우 사각형 표시
            if mouse_data['selecting']:
                start_x = min(mouse_data['selection_start_x'], mouse_data['selection_end_x'])
                start_y = min(mouse_data['selection_start_y'], mouse_data['selection_end_y'])
                end_x = max(mouse_data['selection_start_x'], mouse_data['selection_end_x'])
                end_y = max(mouse_data['selection_start_y'], mouse_data['selection_end_y'])
                
                cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            
        # 프레임 표시
        cv2.imshow('Video', display_frame)
        
        # 키 입력 대기 (일시정지 상태일 때는 더 빠르게 반응하도록)
        wait_time = 1 if is_paused else delay
        key = cv2.waitKey(wait_time) & 0xFF
        
        # 키 입력에 따른 동작
        if key == ord('q'):  # 'q' 키: 종료
            print("사용자가 재생을 중지했습니다.")
            break
            
        elif key == ord('c'):  # 'c' 키: 전체/부분 전환
            if show_full_frame:
                # 중앙 부분을 확대하여 보여주기
                center_x = width // 2
                center_y = height // 2
                crop_size = min(width, height) // 3
                new_crop_region = (center_x - crop_size//2, center_y - crop_size//2, crop_size, crop_size)
                mouse_data['crop_region'] = new_crop_region
                show_full_frame = False
                print(f"자른 영역 보기로 전환: {new_crop_region}")
            else:
                # 전체 프레임 보기로 전환
                new_crop_region = (0, 0, width, height)
                mouse_data['crop_region'] = new_crop_region
                show_full_frame = True
                print("전체 프레임 보기로 전환")
        
        elif key == ord('s'):  # 's' 키: 영역 선택 모드 활성화/비활성화
            mouse_data['selection_mode'] = not mouse_data['selection_mode']
            if mouse_data['selection_mode']:
                print("영역 선택 모드 활성화: 마우스로 드래그하여 영역을 선택하세요.")
            else:
                print("영역 선택 모드 비활성화")
                
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

def process_video(video_path):
    """
    영상을 불러와 프레임 단위로 처리합니다.
    
    Args:
        video_path (str): 영상 파일의 경로
    """
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 열기 실패 시
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    # 동영상의 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"영상 정보:")
    print(f"- 크기: {width}x{height}")
    print(f"- FPS: {fps:.2f}")
    print(f"- 총 프레임 수: {total_frames}")
    
    # 프로세싱 결과를 저장할 경로 설정 (선택사항)
    # output_path = os.path.join(os.path.dirname(video_path), 'processed_' + os.path.basename(video_path))
    # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 여기서 프레임에 원하는 처리 수행
        # 예: 컬러 변환, 크기 조정, 객체 감지 등
        
        # 그레이스케일로 변환 예제
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 블러 적용 예제
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 원본과 처리된 이미지를 나란히 표시
        processed_display = np.hstack([frame, blurred_frame])
        
        # 화면에 맞게 크기 조절 (필요시)
        if processed_display.shape[1] > 1200:
            scale = 1200 / processed_display.shape[1]
            processed_display = cv2.resize(processed_display, None, fx=scale, fy=scale)
        
        # 프레임 번호와 진행상황 표시
        cv2.putText(processed_display, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 화면에 결과 표시
        cv2.imshow('Processing', processed_display)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자가 처리를 중지했습니다.")
            break
        
        # 처리된 프레임 저장 (선택사항)
        # out.write(blurred_frame)
    
    # 처리 종료 및 통계 출력
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps_processing = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n처리 완료:")
    print(f"- 처리된 프레임: {frame_count}")
    print(f"- 소요 시간: {elapsed_time:.2f}초")
    print(f"- 처리 속도: {fps_processing:.2f} FPS")
    
    # 자원 해제
    cap.release()
    # out.release()  # 주석 해제 시 처리 영상 저장
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 직접 영상 경로 지정
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4"
    
    # 파일 존재 확인
    if os.path.exists(video_path):
        print(f"영상 파일을 로딩합니다: {os.path.basename(video_path)}")
        
        # 영상 재생 (기본 설정: 0.3배 크기로 축소하여 재생)
        play_video(video_path, scale_factor=0.3, crop_region=None)
    else:
        print(f"파일이 존재하지 않습니다: {video_path}")
