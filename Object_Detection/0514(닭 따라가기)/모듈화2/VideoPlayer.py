from TextRenderer import TextRenderer
from ChickenDetector import ChickenDetector

class VideoPlayer:
    """그리드 분할과 닭 탐지 기능을 갖춘 비디오 플레이어 클래스"""
    
    def __init__(self, video_path, model_path, grid_size=5, scale_factor=1.0):
        """
        VideoPlayer 초기화
        
        Args:
            video_path (str): 영상 파일의 경로
            model_path (str): YOLO 모델 파일 경로
            grid_size (int): 분할 그리드 크기 (grid_size x grid_size)
            scale_factor (float): 영상 크기 조절 비율 (기본값: 1.0)
        """
        self.video_path = video_path
        self.grid_size = grid_size
        self.scale_factor = scale_factor
        
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
        self.delay = int(1000 / self.fps)  # 프레임 간 지연시간 (밀리초)
        
        # 비디오 정보 출력
        self._print_video_info()
        return True
    
    def _print_video_info(self):
        """비디오 정보와 조작 방법을 출력합니다."""
        print(f"원본 영상 크기: {self.width}x{self.height}")
        print(f"조절 비율: {self.scale_factor}")
        print(f"영상 재생 중... (FPS: {self.fps:.2f})")
        print(f"그리드 크기: {self.grid_size}x{self.grid_size}")
        print("조작 방법:")
        print("- 'q' 키: 종료")
        print("- 'g' 키: 그리드 모드/확대 모드 전환")
        print("- '+/-' 키: 확대/축소")
        print("- 'y' 키: YOLO 탐지 켜기/끄기")
        print("- '[/]' 키: 탐지 임계값 조절")
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
    
    def detect_chickens(self, frame):
        """
        프레임에서 닭을 탐지합니다.
        
        Args:
            frame: 탐지할 프레임
            
        Returns:
            처리된 프레임과 닭 개수
        """
        chicken_count = 0
        
        if self.detector.enabled and self.yolo_detection_active:
            if not self.is_grid_mode and self.selected_cell is not None:
                # 선택된 셀에서만 닭 탐지
                row, col = self.selected_cell
                cell_height = self.height // self.grid_size
                cell_width = self.width // self.grid_size
                x = col * cell_width
                y = row * cell_height
                cell_frame = frame[y:y+cell_height, x:x+cell_width]
                results, chicken_count = self.detector.detect(cell_frame, self.conf_threshold)
                
                # 탐지 결과를 프레임에 그림 (레이블 크기와 선 두께 조정)
                if results is not None:
                    plotted_cell = results.plot(labels=True, font_size=0.5, line_width=1)
                    frame[y:y+cell_height, x:x+cell_width] = plotted_cell
            else:
                # 전체 프레임에서 닭 탐지
                results, chicken_count = self.detector.detect(frame, self.conf_threshold)
                
                # 탐지 결과를 프레임에 그림 (레이블 크기와 선 두께 조정)
                if results is not None:
                    frame = results.plot(labels=True, font_size=0.4, line_width=1)
        
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
        
        # 상태 정보 표시 (한글 지원)
        frame = self.text_renderer.put_text(frame, f"배율: {self.scale_factor:.2f}", (10, 30), 25, (0, 255, 0))
        frame = self.text_renderer.put_text(frame, f"시간: {time_pos:.1f}초 / {total_time:.1f}초", (10, 60), 25, (0, 255, 0))
        frame = self.text_renderer.put_text(frame, f"{'일시정지' if self.is_paused else '재생 중'}", (10, 90), 25, (0, 255, 0))
        frame = self.text_renderer.put_text(frame, f"모드: {'그리드' if self.is_grid_mode else '확대'}", (10, 120), 25, (0, 255, 0))
        
        # 선택된 셀 정보 표시
        if not self.is_grid_mode and self.selected_cell is not None:
            frame = self.text_renderer.put_text(frame, f"선택 셀: ({self.selected_cell[0]}, {self.selected_cell[1]})", 
                    (10, 150), 25, (0, 255, 0))
        
        # YOLO 탐지 정보 표시
        if self.detector.enabled:
            color = (0, 255, 255) if self.yolo_detection_active else (0, 0, 255)
            frame = self.text_renderer.put_text(frame, 
                    f"YOLO: {'켜짐' if self.yolo_detection_active else '꺼짐'} (임계값: {self.conf_threshold:.2f})", 
                    (10, 180), 25, color)
            
            if self.yolo_detection_active:
                frame = self.text_renderer.put_text(frame, f"탐지된 닭: {chicken_count}마리", 
                        (10, 210), 25, (0, 255, 255))
        
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
