import cv2
import os

from TextRenderer import TextRenderer
from ChickenDetector import ChickenDetector

class VideoPlayer:
    """그리드 분할과 닭 탐지 기능을 갖춘 비디오 플레이어 클래스"""
    def __init__(self, video_path, model_path, grid_size=5, scale_factor=1.0, detection_interval=5):
        """
        VideoPlayer 초기화
        
        Args:
            video_path (str): 영상 파일의 경로
            model_path (str): YOLO 모델 파일 경로
            grid_size (int): 분할 그리드 크기 (grid_size x grid_size)
            scale_factor (float): 영상 크기 조절 비율 (기본값: 1.0)
            detection_interval (int): 몇 프레임마다 객체 탐지를 수행할지 설정 (기본값: 5)
        """
        self.video_path = video_path
        self.grid_size = grid_size
        self.scale_factor = scale_factor
        self.detection_interval = detection_interval
        self.frame_count = 0  # 프레임 카운터
        
        # 비디오 캡처 객체 생성
        self.cap = None
        
        # 닭 탐지 객체
        self.detector = ChickenDetector(model_path)
        
        # 텍스트 렌더러
        self.text_renderer = TextRenderer()
        
        # 상태 변수 초기화
        self.is_paused = False
        self.curr_frame_pos = 0
        self.is_grid_mode = True
        self.selected_cell = None
        self.yolo_detection_active = False
        self.conf_threshold = 0.5
        self.latest_results = None  # 가장 최근 탐지 결과 저장
        
        # 비디오 정보 변수
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0
        self.delay = 0
        
    def initialize_video(self):
        """비디오 파일을 열고 초기화합니다."""
        if not os.path.exists(self.video_path):
            print(f"파일이 존재하지 않습니다: {self.video_path}")
            return False
            
        # 비디오 캡처 객체 생성
        self.cap = cv2.VideoCapture(self.video_path)
        
        # 비디오 열기 실패 시
        if not self.cap.isOpened():
            print(f"비디오 파일을 열 수 없습니다: {self.video_path}")
            return False
        
        # 비디오의 정보 가져오기
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.delay = int(1000 / self.fps)  # 프레임 간 지연시간 (밀리초)        # 비디오 정보 출력
        self._print_video_info()
        return True
        
    def _print_video_info(self):
        """비디오 정보와 조작 방법을 출력합니다."""
        print(f"원본 영상 크기: {self.width}x{self.height}")
        print(f"조절 비율: {self.scale_factor}")
        print(f"영상 재생 중... (FPS: {self.fps:.2f})")
        print(f"그리드 크기: {self.grid_size}x{self.grid_size}")
        print(f"객체 탐지 간격: {self.detection_interval}프레임")
        print("조작 방법:")
        print("- 'q' 키: 종료")
        print("- 'g' 키: 그리드 모드/확대 모드 전환")
        print("- '+/-' 키: 확대/축소")
        print("- 'y' 키: YOLO 탐지 켜기/끄기")
        print("- 't' 키: 객체 추적(ID 표시) 켜기/끄기")
        print("- 'c' 키: 일관된 ID 추적 켜기/끄기")
        print("- '[/]' 키: 탐지 임계값 조절")
        print("- '1'/'2' 키: 탐지 간격 감소/증가")
        print("- 스페이스바: 재생/일시정지")
        print("- 'a'/'d' 키: 뒤로/앞으로 5초")
        print("- 마우스 클릭: 그리드 칸 선택/확대")
    
    def setup_mouse_callback(self):
        """마우스 콜백 함수를 설정합니다."""
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and self.is_grid_mode:
                # 그리드 모드에서 마우스 클릭 시 해당 셀 선택
                col = int(x / (self.width * self.scale_factor / self.grid_size))
                row = int(y / (self.height * self.scale_factor / self.grid_size))
                
                # 그리드 범위 내의 셀만 선택 가능
                if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
                    self.selected_cell = (row, col)
                    self.is_grid_mode = False
                    print(f"선택한 그리드 셀: ({row}, {col})")
        
        # 마우스 콜백 등록
        cv2.namedWindow('Video')
        cv2.setMouseCallback('Video', mouse_callback)
    
    def read_frame(self):
        """다음 프레임을 읽거나 현재 프레임을 유지합니다."""
        if not self.is_paused:
            # 일반 재생 모드: 다음 프레임 읽기
            ret, frame = self.cap.read()
            self.curr_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            # 일시정지 모드: 현재 프레임 유지
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame_pos - 1)
            ret, frame = self.cap.read()
            
        return ret, frame

    def _get_id_set_for_detection(self, results, i, original_id):
        """
        ID가 속한 집합을 가져옵니다.
        
        Args:
            results: 탐지 결과
            i: 현재 객체 인덱스
            original_id: 원래 ID
            
        Returns:
            ID가 속한 집합
        """
        id_set = set()
        
        # 결과에서 직접 ID 세트 가져오기 (이미 계산된 세트 사용)
        if hasattr(results, 'id_sets') and i < len(results.id_sets):
            id_set = results.id_sets[i]
        # 백업: detector에서 직접 가져오기
        elif hasattr(self.detector, 'id_to_set_index') and original_id in self.detector.id_to_set_index:
            set_index = self.detector.id_to_set_index[original_id]
            if set_index < len(self.detector.chicken_id_sets):
                id_set = self.detector.chicken_id_sets[set_index]
                
        return id_set
    
    def _draw_id_text_with_background(self, frame, text, x1, y1, font_scale=0.7, color=(0, 255, 255)):
        """
        ID 텍스트를 배경과 함께 그립니다.
        
        Args:
            frame: 그릴 프레임
            text: 표시할 텍스트
            x1, y1: 박스 좌표
            font_scale: 폰트 크기
            color: 텍스트 색상
            
        Returns:
            그려진 프레임
        """
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 텍스트 위치 계산 (상단에 표시)
        text_bg_top = max(0, int(y1) - text_height - 12)
        
        # 더 명확한 배경 - 반투명 검은색 배경에 테두리 추가
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                    (int(x1) - 2, text_bg_top - 2), 
                    (int(x1) + text_width + 4, text_bg_top + text_height + 10), 
                    (0, 0, 0), -1)
        cv2.rectangle(overlay, 
                    (int(x1) - 2, text_bg_top - 2), 
                    (int(x1) + text_width + 4, text_bg_top + text_height + 10), 
                    (255, 255, 0), 1)  # 노란색 테두리
        
        # 이미지에 오버레이 적용 (80% 불투명도)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # 텍스트 그리기 - 더 밝은 색상으로
        cv2.putText(frame, text, (int(x1), int(y1) - 7),
                font, font_scale, color, thickness)
        
        return frame

    def detect_chickens(self, frame):
        """
        프레임에서 닭을 탐지합니다.
        설정된 프레임 간격(detection_interval)에 따라 객체 탐지를 수행합니다.
        
        Args:
            frame: 탐지할 프레임
            
        Returns:
            처리된 프레임과 닭 개수
        """
        chicken_count = 0
        
        # 프레임 카운터 증가
        self.frame_count = (self.frame_count + 1) % self.detection_interval
        
        # YOLO 탐지가 활성화되고, 현재 프레임이 탐지 간격에 해당하는 경우에만 탐지 수행
        do_detection = self.frame_count == 0 or self.is_paused
        
        if self.detector.enabled and self.yolo_detection_active:
            if not self.is_grid_mode and self.selected_cell is not None:
                # 선택된 셀에서만 닭 탐지
                row, col = self.selected_cell
                cell_height = self.height // self.grid_size
                cell_width = self.width // self.grid_size
                x = col * cell_width
                y = row * cell_height
                cell_frame = frame[y:y+cell_height, x:x+cell_width]
                
                # 객체 탐지 간격에 따른 탐지
                if do_detection:
                    results, chicken_count = self.detector.detect(cell_frame, self.conf_threshold)
                    self.latest_results = results
                else:
                    # 이전 탐지 결과 사용
                    results = self.latest_results
                    if results is not None:
                        chicken_count = len(results.boxes) if hasattr(results, 'boxes') else 0
                  # 탐지 결과를 프레임에 그림 (레이블 크기와 선 두께 조정)
                if results is not None:                    # 기본 시각화
                    plotted_cell = results.plot(labels=True, font_size=0.5, line_width=1)
                    
                    # 일관된 ID 표시가 활성화된 경우 ID를 수동으로 오버레이
                    if hasattr(self.detector, 'use_consistent_ids') and self.detector.use_consistent_ids and hasattr(results, 'display_ids'):
                        # 박스와 ID 정보 가져오기
                        xyxy = results.boxes.xyxy.cpu().numpy()
                        
                        for i, box in enumerate(xyxy):
                            if i < len(results.display_ids):                                # 박스 좌표
                                x1, y1, x2, y2 = box
                                # 원래 ID와 일관된 ID 모두 표시
                                original_id = int(results.boxes.id[i].item())
                                consistent_id = results.display_ids[i]
                                
                                # ID가 속한 집합 가져오기
                                id_set = self._get_id_set_for_detection(results, i, original_id)
                                if id_set:
                                    # 간결한 ID 세트 표시 형식으로 수정
                                    id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                                    text = f"ID:{consistent_id} {id_set_str}"
                                    plotted_cell = self._draw_id_text_with_background(plotted_cell, text, x1, y1)
                                elif original_id != consistent_id:
                                    # 매핑 정보만 있는 경우
                                    text = f"ID:{original_id}->{consistent_id}"
                                    plotted_cell = self._draw_id_text_with_background(plotted_cell, text, x1, y1, font_scale=0.6, color=(0, 0, 255))
                                else:
                                    # 변경이 없는 경우 일관된 ID만 표시
                                    text = f"ID:{consistent_id}"
                                    plotted_cell = self._draw_id_text_with_background(plotted_cell, text, x1, y1, font_scale=0.6, color=(0, 0, 255))
                    
                    frame[y:y+cell_height, x:x+cell_width] = plotted_cell
            else:
                # 전체 프레임에서 닭 탐지
                if do_detection:
                    results, chicken_count = self.detector.detect(frame, self.conf_threshold)
                    self.latest_results = results
                else:                    # 이전 탐지 결과 사용
                    results = self.latest_results
                    if results is not None:
                        chicken_count = len(results.boxes) if hasattr(results, 'boxes') else 0
                
                # 탐지 결과를 프레임에 그림 (레이블 크기와 선 두께 조정)
                if results is not None:
                    # 기본 시각화
                    frame = results.plot(labels=True, font_size=0.4, line_width=1)
                    
                    # 일관된 ID 표시가 활성화된 경우 ID를 수동으로 오버레이
                    if hasattr(self.detector, 'use_consistent_ids') and self.detector.use_consistent_ids and hasattr(results, 'display_ids'):
                        # 박스와 ID 정보 가져오기
                        xyxy = results.boxes.xyxy.cpu().numpy()
                        
                        for i, box in enumerate(xyxy):
                            if i < len(results.display_ids):                                # 박스 좌표
                                x1, y1, x2, y2 = box
                                # 원래 ID와 일관된 ID 모두 표시
                                original_id = int(results.boxes.id[i].item())
                                consistent_id = results.display_ids[i]
                                
                                # ID가 속한 집합 가져오기
                                id_set = self._get_id_set_for_detection(results, i, original_id)
                                if id_set:
                                    # 간결한 ID 세트 표시 형식으로 수정
                                    id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                                    text = f"ID:{consistent_id} {id_set_str}"
                                    frame = self._draw_id_text_with_background(frame, text, x1, y1)
                                elif original_id != consistent_id:
                                    # 매핑 정보만 있는 경우
                                    text = f"ID:{original_id}->{consistent_id}"
                                    frame = self._draw_id_text_with_background(frame, text, x1, y1, font_scale=0.6, color=(0, 0, 255))
                                else:
                                    # 변경이 없는 경우 일관된 ID만 표시
                                    text = f"ID:{consistent_id}"
                                    frame = self._draw_id_text_with_background(frame, text, x1, y1, font_scale=0.6, color=(0, 0, 255))
        
        return frame, chicken_count
    
    def prepare_display_frame(self, frame, chicken_count):
        """
        화면 표시를 위한 프레임을 준비합니다.
        
        Args:
            frame: 원본 프레임
            chicken_count: 탐지된 닭의 수
            
        Returns:
            표시용 프레임
        """
        # 그리드 모드 또는 확대 모드에 따라 표시 프레임 준비
        if self.is_grid_mode:
            display_frame = self._prepare_grid_mode_frame(frame)
        else:
            display_frame = self._prepare_zoomed_mode_frame(frame)
        
        # 크기 조절
        if self.scale_factor != 1.0:
            display_height, display_width = display_frame.shape[:2]
            new_width = int(display_width * self.scale_factor)
            new_height = int(display_height * self.scale_factor)
            display_frame = cv2.resize(display_frame, (new_width, new_height))
          # 상태 정보 표시
        display_frame = self._add_status_info(display_frame, chicken_count)
        
        return display_frame
    
    def _prepare_grid_mode_frame(self, frame):
        """그리드 모드에서의 프레임을 준비합니다."""
        display_frame = frame.copy()
        frame_height, frame_width = display_frame.shape[:2]
        
        # 그리드 선 그리기
        cell_height = frame_height // self.grid_size
        cell_width = frame_width // self.grid_size
        
        # 가로선 그리기
        for i in range(1, self.grid_size):
            y = i * cell_height
            cv2.line(display_frame, (0, y), (frame_width, y), (0, 255, 0), 2)
        
        # 세로선 그리기
        for i in range(1, self.grid_size):
            x = i * cell_width
            cv2.line(display_frame, (x, 0), (x, frame_height), (0, 255, 0), 2)
        
        # 각 셀에 번호 표시
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * cell_width + cell_width // 2 - 20
                y = row * cell_height + cell_height // 2
                cv2.putText(display_frame, f"({row},{col})", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return display_frame
    def _prepare_zoomed_mode_frame(self, frame):
        """확대 모드에서의 프레임을 준비합니다."""
        if self.selected_cell is not None:
            row, col = self.selected_cell
            cell_height = self.height // self.grid_size
            cell_width = self.width // self.grid_size
            x = col * cell_width
            y = row * cell_height
            display_frame = frame[y:y+cell_height, x:x+cell_width].copy()
        else:
            display_frame = frame.copy()
            self.is_grid_mode = True  # 선택된 셀이 없으면 그리드 모드로 전환
        
        return display_frame
    def _add_status_info(self, frame, chicken_count):
        """프레임에 상태 정보를 추가합니다."""
        time_pos = self.curr_frame_pos / self.fps  # 현재 시간 위치(초)
        total_time = self.total_frames / self.fps   # 총 시간(초)
        
        # 상태 정보 표시 (한글 지원) - 배경 추가로 가독성 향상
        frame = self.text_renderer.put_text(frame, f"배율: {self.scale_factor:.2f}", 
                (10, 30), 25, (0, 255, 0), with_background=True)
        frame = self.text_renderer.put_text(frame, f"시간: {time_pos:.1f}초 / {total_time:.1f}초", 
                (10, 60), 25, (0, 255, 0), with_background=True)
        frame = self.text_renderer.put_text(frame, f"{'일시정지' if self.is_paused else '재생 중'}", 
                (10, 90), 25, (0, 255, 0), with_background=True)
        frame = self.text_renderer.put_text(frame, f"모드: {'그리드' if self.is_grid_mode else '확대'}", 
                (10, 120), 25, (0, 255, 0), with_background=True)
        
        # 선택된 셀 정보 표시
        if not self.is_grid_mode and self.selected_cell is not None:
            frame = self.text_renderer.put_text(frame, f"선택 셀: ({self.selected_cell[0]}, {self.selected_cell[1]})", 
                    (10, 150), 25, (0, 255, 0), with_background=True)
        
        # YOLO 탐지 정보 표시
        if self.detector.enabled:
            color = (0, 255, 255) if self.yolo_detection_active else (0, 0, 255)
            frame = self.text_renderer.put_text(frame, 
                    f"YOLO: {'켜짐' if self.yolo_detection_active else '꺼짐'} (임계값: {self.conf_threshold:.2f})", 
                    (10, 180), 25, color, with_background=True)
            
            if self.yolo_detection_active:
                frame = self.text_renderer.put_text(frame, f"탐지된 닭: {chicken_count}마리", 
                        (10, 210), 25, (0, 255, 255), with_background=True)
                frame = self.text_renderer.put_text(frame, f"탐지 간격: {self.detection_interval}프레임마다", 
                        (10, 240), 25, (0, 255, 255), with_background=True)
                  # 추적 상태 표시
                tracking_color = (0, 255, 255) if hasattr(self.detector, 'tracking_enabled') and self.detector.tracking_enabled else (0, 0, 255)
                frame = self.text_renderer.put_text(frame, 
                        f"객체 추적: {'켜짐' if hasattr(self.detector, 'tracking_enabled') and self.detector.tracking_enabled else '꺼짐'}", 
                        (10, 270), 25, tracking_color, with_background=True)
                
                # 일관된 ID 추적 상태 표시
                if hasattr(self.detector, 'tracking_enabled') and self.detector.tracking_enabled:
                    consistent_id_color = (0, 255, 255) if hasattr(self.detector, 'use_consistent_ids') and self.detector.use_consistent_ids else (0, 0, 255)
                    frame = self.text_renderer.put_text(frame, 
                            f"일관된 ID: {'켜짐' if hasattr(self.detector, 'use_consistent_ids') and self.detector.use_consistent_ids else '꺼짐'}", 
                            (10, 300), 25, consistent_id_color, with_background=True)
        
        return frame    
    def _visualize_id_sets(self, frame, x_offset=10, y_offset=330):
        """
        객체 ID 집합을 화면에 시각화합니다.
        
        Args:
            frame: 시각화할 프레임
            x_offset: X 좌표 시작점
            y_offset: Y 좌표 시작점
            
        Returns:
            시각화가 추가된 프레임
        """
        # 이제 ID 집합 정보는 각 객체 위에 직접 표시되므로 이 함수는 간단한 상태 정보만 표시
        if not hasattr(self.detector, 'use_consistent_ids') or not self.detector.use_consistent_ids:
            return frame
            
        # ID 집합 정보가 있는 경우, 현재 활성 ID 세트 개수만 표시
        if hasattr(self.detector, 'chicken_id_sets') and len(self.detector.chicken_id_sets) > 0:
            # 빈 집합 제외한 세트 개수 계산
            active_sets = sum(1 for s in self.detector.chicken_id_sets if s)
            
            # 활성 ID 세트 개수 표시
            msg = f"활성 ID 세트: {active_sets}개"
            frame = self.text_renderer.put_text(frame, msg, 
                    (x_offset, y_offset), 25, (255, 255, 0), with_background=True)
        
        return frame

    def handle_keypress(self, key):
        """
        키 입력에 따른 동작을 처리합니다.
        
        Args:
            key: 입력된 키 코드
            
        Returns:
            계속 실행 여부 (False면 종료)
        """
        if key == ord('q'):  # 'q' 키: 종료
            print("사용자가 재생을 중지했습니다.")
            return False
        
        elif key == ord('g'):  # 'g' 키: 그리드 모드/확대 모드 전환
            self.is_grid_mode = not self.is_grid_mode
            print(f"{'그리드' if self.is_grid_mode else '확대'} 모드로 전환")
        
        elif key == ord('y'):  # 'y' 키: YOLO 탐지 켜기/끄기
            if self.detector.enabled:
                self.yolo_detection_active = not self.yolo_detection_active
                print(f"YOLO 탐지: {'활성화' if self.yolo_detection_active else '비활성화'}")
            else:                
                print("YOLO 모델이 로드되지 않았습니다.")
        
        elif key == ord('t'):  # 't' 키: 객체 추적 켜기/끄기
            if self.detector.enabled and hasattr(self.detector, 'toggle_tracking'):
                self.detector.toggle_tracking()
                
        elif key == ord('c'):  # 'c' 키: 일관된 ID 추적 켜기/끄기
            if self.detector.enabled and hasattr(self.detector, 'toggle_consistent_ids'):
                self.detector.toggle_consistent_ids()
                
        elif key == ord('1'):  # '1' 키: 객체 탐지 간격 감소
            if self.detection_interval > 1:
                self.detection_interval -= 1
                print(f"객체 탐지 간격 변경: {self.detection_interval}프레임마다")
            
        elif key == ord('2'):  # '2' 키: 객체 탐지 간격 증가
            self.detection_interval += 1
            print(f"객체 탐지 간격 변경: {self.detection_interval}프레임마다")
            
        elif key == ord('['):  # '[' 키: 임계값 낮추기
            if self.detector.enabled:
                self.conf_threshold = max(0.1, self.conf_threshold - 0.1)
                print(f"탐지 임계값 변경: {self.conf_threshold:.2f}")
                
        elif key == ord(']'):  # ']' 키: 임계값 높이기
            if self.detector.enabled:
                self.conf_threshold = min(0.9, self.conf_threshold + 0.1)
                print(f"탐지 임계값 변경: {self.conf_threshold:.2f}")
                
        elif key == ord('+') or key == ord('='):  # '+' 키: 확대
            self.scale_factor += 0.1
            print(f"확대: {self.scale_factor:.2f}")
            
        elif key == ord('-'):  # '-' 키: 축소
            if self.scale_factor > 0.2:  # 너무 작아지지 않도록 제한
                self.scale_factor -= 0.1
                print(f"축소: {self.scale_factor:.2f}")
                
        elif key == 32:  # 스페이스바: 재생/일시정지
            self.is_paused = not self.is_paused
            print("일시정지" if self.is_paused else "재생")
            
        elif key == ord('d'):  # d키: 앞으로 이동
            # 5초 앞으로
            target_frame = min(self.curr_frame_pos + int(self.fps * 5), self.total_frames - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.curr_frame_pos = target_frame
            print(f"앞으로 이동: {target_frame / self.fps:.1f}초")
            
        elif key == ord('a'):  # a키: 뒤로 이동
            # 5초 뒤로
            target_frame = max(self.curr_frame_pos - int(self.fps * 5), 0)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.curr_frame_pos = target_frame
            print(f"뒤로 이동: {target_frame / self.fps:.1f}초")
        
        return True
    
    def play(self):
        """비디오를 재생하고 인터랙션을 처리합니다."""
        # 비디오 초기화 및 마우스 콜백 설정
        if not self.initialize_video():
            return
            
        self.setup_mouse_callback()
        
        # 메인 루프
        while True:
            # 프레임 읽기
            ret, frame = self.read_frame()
            
            if not ret:
                print("영상이 끝났거나 읽기 실패.")
                break
            
            # 닭 탐지
            frame, chicken_count = self.detect_chickens(frame)
            
            # 화면 표시 준비
            display_frame = self.prepare_display_frame(frame, chicken_count)
            
            # 프레임 표시
            cv2.imshow('Video', display_frame)
            
            # 키 입력 대기 (일시정지 상태일 때는 더 빠르게 반응하도록)
            wait_time = 1 if self.is_paused else self.delay
            key = cv2.waitKey(wait_time) & 0xFF
            
            # 키 입력에 따른 동작
            if not self.handle_keypress(key):
                break
        
        # 자원 해제
        self.cap.release()
        cv2.destroyAllWindows()
