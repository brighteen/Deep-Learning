import cv2
import os
import numpy as np
from ChickenDetector import ChickenDetector

class VideoPlayer:
    """ID 관리에 중점을 둔 간소화된 비디오 플레이어 클래스"""
    
    def __init__(self, video_path, model_path, scale_factor=1.0, detection_interval=10):
        """
        VideoPlayer 초기화
        
        Args:
            video_path (str): 영상 파일의 경로
            model_path (str): YOLO 모델 파일 경로
            scale_factor (float): 영상 크기 조절 비율 (기본값: 1.0)
            detection_interval (int): 몇 프레임마다 객체 탐지를 수행할지 설정 (기본값: 10)
        """
        self.video_path = video_path
        self.scale_factor = scale_factor
        self.detection_interval = detection_interval
        self.frame_count = 0  # 프레임 카운터
        
        # 비디오 캡처 객체 생성
        self.cap = None
        
        # 닭 탐지 객체
        self.detector = ChickenDetector(model_path)
        
        # ROI(관심 영역) 변수
        self.roi_x1 = 0  # ROI 시작 X 좌표
        self.roi_y1 = 0  # ROI 시작 Y 좌표
        self.roi_x2 = None  # ROI 끝 X 좌표 (None이면 전체 너비 사용)
        self.roi_y2 = None  # ROI 끝 Y 좌표 (None이면 전체 높이 사용)
        self.curr_frame_pos = 0
        self.conf_threshold = 0.5
        self.latest_results = None  # 가장 최근 탐지 결과 저장
        
        # 비디오 정보 변수
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0
        self.delay = 0
        
        # 영상 분석 범위 설정
        self.start_time = 0  # 시작 시간(초)
        self.end_time = None  # 종료 시간(초), None이면 끝까지
        self.start_frame = 0
        self.end_frame = None  # None이면 마지막 프레임까지
        
        # 프레임 영역 설정
        self.roi_x1 = 0
        self.roi_y1 = 0
        self.roi_x2 = None  # None이면 전체 너비 사용
        self.roi_y2 = None  # None이면 전체 높이 사용
        
    def set_video_range(self, start_time=0, end_time=None):
        """
        영상 분석 범위를 설정합니다.
        
        Args:
            start_time (float): 시작 시간(초)
            end_time (float): 종료 시간(초), None이면 영상 끝까지
        """
        if not self.cap:
            print("비디오가 초기화되지 않았습니다. initialize_video()를 먼저 호출하세요.")
            return
            
        self.start_time = max(0, start_time)
        self.start_frame = int(self.start_time * self.fps)
        
        if end_time is not None:
            self.end_time = min(end_time, self.total_frames / self.fps)
            self.end_frame = int(self.end_time * self.fps)
        else:
            self.end_time = None
            self.end_frame = None
        
        # 시작 위치로 이동
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.curr_frame_pos = self.start_frame
        
        print(f"영상 분석 범위 설정: {self.start_time:.1f}초 ~ {self.end_time if self.end_time is not None else '끝'}초")
        print(f"프레임 범위: {self.start_frame} ~ {self.end_frame if self.end_frame is not None else '끝'}")
        
    def set_frame_roi(self, x1=0, y1=0, x2=None, y2=None):
        """
        프레임 내 관심 영역(ROI)을 설정합니다.
        
        Args:
            x1 (int): 시작 x 좌표
            y1 (int): 시작 y 좌표
            x2 (int): 종료 x 좌표 (None이면 전체 너비 사용)
            y2 (int): 종료 y 좌표 (None이면 전체 높이 사용)
        """
        if not self.cap:
            print("비디오가 초기화되지 않았습니다. initialize_video()를 먼저 호출하세요.")
            return
            
        self.roi_x1 = max(0, x1)
        self.roi_y1 = max(0, y1)
        
        if x2 is not None:
            self.roi_x2 = min(x2, self.width)
        else:
            self.roi_x2 = self.width
            
        if y2 is not None:
            self.roi_y2 = min(y2, self.height)
        else:
            self.roi_y2 = self.height
            
        print(f"프레임 ROI 설정: ({self.roi_x1}, {self.roi_y1}) ~ ({self.roi_x2}, {self.roi_y2})")
        print(f"ROI 크기: {self.roi_x2 - self.roi_x1}x{self.roi_y2 - self.roi_y1}")
        
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
        self.delay = int(1000 / self.fps)
        
        # 비디오 정보 출력
        print(f"원본 영상 크기: {self.width}x{self.height}")
        print(f"조절 비율: {self.scale_factor}")
        print(f"영상 재생 중... (FPS: {self.fps:.2f})")
        print(f"총 프레임 수: {self.total_frames}, 총 재생 시간: {self.total_frames/self.fps:.1f}초")
        print(f"객체 탐지 간격: {self.detection_interval}프레임")
        print("조작 방법:")
        print("- ESC 키: 종료")
        print("- 'a' 키: 5초 뒤로 이동")
        print("- 'd' 키: 5초 앞으로 이동")
        return True
        
    def read_frame(self):
        """다음 프레임을 읽습니다."""
        # 프레임 읽기
        ret, frame = self.cap.read()
        if ret:
            self.curr_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 종료 프레임에 도달했는지 확인
            if self.end_frame is not None and self.curr_frame_pos >= self.end_frame:
                print(f"설정된 종료 시간({self.end_time:.1f}초)에 도달했습니다.")
                return False, None
                
            # 프레임을 성공적으로 읽었을 경우에만 ROI 적용
            if frame is not None:
                # 설정된 ROI 영역으로 프레임 자르기
                frame = frame[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        
        return ret, frame

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
        
        # 반투명 검은색 배경에 테두리 추가
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
        
        # 텍스트 그리기
        cv2.putText(frame, text, (int(x1), int(y1) - 7),
                font, font_scale, color, thickness)
        
        return frame

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
        
        # 결과에서 직접 ID 집합 가져오기
        if hasattr(results, 'id_sets') and i < len(results.id_sets):
            id_set = results.id_sets[i]
        # 백업: detector에서 직접 가져오기
        elif hasattr(self.detector, 'id_to_set_index') and original_id in self.detector.id_to_set_index:
            set_index = self.detector.id_to_set_index[original_id]
            if set_index < len(self.detector.chicken_id_sets):
                id_set = self.detector.chicken_id_sets[set_index]
                
        return id_set

    def detect_chickens(self, frame):
        """
        프레임에서 닭을 탐지합니다.
        
        Args:
            frame: 탐지할 프레임
            
        Returns:
            처리된 프레임과 닭 개수
        """
        chicken_count = 0
        
        # 프레임 카운터 증가
        self.frame_count = (self.frame_count + 1) % self.detection_interval
        
        # 탐지 수행 여부 결정
        do_detection = self.frame_count == 0
        
        # 객체 탐지 수행
        if do_detection:
            results, chicken_count = self.detector.detect(frame, self.conf_threshold)
            self.latest_results = results
        else:
            # 이전 탐지 결과 사용
            results = self.latest_results
            if results is not None:
                chicken_count = len(results.boxes) if hasattr(results, 'boxes') else 0
        
        # 탐지 결과를 프레임에 그림
        if results is not None:
            # 기본 시각화
            frame = results.plot(labels=True, font_size=0.4, line_width=1)
            
            # ID를 수동으로 오버레이
            if hasattr(results, 'display_ids') and hasattr(results.boxes, 'xyxy'):
                # 박스와 ID 정보 가져오기
                xyxy = results.boxes.xyxy.cpu().numpy()
                
                for i, box in enumerate(xyxy):
                    if i < len(results.display_ids):
                        # 박스 좌표
                        x1, y1, x2, y2 = box
                        # 원래 ID와 일관된 ID
                        original_id = int(results.boxes.id[i].item())
                        consistent_id = results.display_ids[i]
                        
                        # ID가 속한 집합 가져오기
                        id_set = self._get_id_set_for_detection(results, i, original_id)
                        if id_set:
                            # ID 세트 표시
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
    
    def _add_status_info(self, frame, chicken_count):
        """프레임에 상태 정보를 추가합니다."""
        time_pos = self.curr_frame_pos / self.fps  # 현재 시간 위치(초)
        total_time = self.total_frames / self.fps   # 총 시간(초)
        
        # 상태 정보 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"시간: {time_pos:.1f}초 / {total_time:.1f}초", 
                   (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"탐지된 닭: {chicken_count}마리", 
                   (10, 60), font, 0.7, (0, 255, 255), 2)
                
        # ID 집합 개수 표시
        if hasattr(self.detector, 'chicken_id_sets'):
            active_sets = sum(1 for s in self.detector.chicken_id_sets if s)
            cv2.putText(frame, f"활성 ID 집합: {active_sets}개", 
                       (10, 90), font, 0.7, (255, 255, 0), 2)
        
        return frame
        
    def handle_keypress(self, key):
        """
        키 입력에 따른 동작을 처리합니다.
        
        Args:
            key: 입력된 키 코드
            
        Returns:
            계속 실행 여부 (False면 종료)
        """
        if key == 27:  # ESC 키: 종료
            print("프로그램을 종료합니다.")
            return False
            
        elif key == ord('d'):  # d키: 앞으로 이동
            # 5초 앞으로
            target_frame = min(self.curr_frame_pos + int(self.fps * 5), 
                               self.end_frame if self.end_frame is not None else self.total_frames - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.curr_frame_pos = target_frame
            print(f"5초 앞으로 이동: {target_frame / self.fps:.1f}초")
            
        elif key == ord('a'):  # a키: 뒤로 이동
            # 5초 뒤로
            target_frame = max(self.curr_frame_pos - int(self.fps * 5), 
                               self.start_frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.curr_frame_pos = target_frame
            print(f"5초 뒤로 이동: {target_frame / self.fps:.1f}초")
        
        return True
    
    def set_playback_speed(self, speed_factor=1.0):
        """
        영상 재생 속도를 설정합니다.
        
        Args:
            speed_factor (float): 속도 배율 (1.0이 기본 속도, 2.0이면 2배 빠르게)
        """
        if speed_factor <= 0:
            print("속도 배율은 0보다 커야 합니다.")
            return
            
        self.delay = int(1000 / (self.fps * speed_factor))
        print(f"재생 속도 설정: {speed_factor}x (딜레이: {self.delay}ms)")
        
    def play(self):
        """비디오를 재생하고 ID 추적 결과를 표시합니다."""
        # 비디오 초기화
        if not self.initialize_video():
            return
            
        # 비디오 범위가 설정되지 않은 경우, 기본값 사용
        if self.start_frame == 0 and self.end_frame is None:
            self.set_video_range(0)
            
        # 윈도우 생성 및 속성 설정
        cv2.namedWindow('Chicken Tracking', cv2.WINDOW_NORMAL)
            
        # 성능 측정을 위한 변수
        frame_time_start = cv2.getTickCount()
        processed_frames = 0
        fps_update_interval = 30  # 30프레임마다 FPS 계산

        # 메인 루프
        while True:
            # 프레임 읽기
            ret, frame = self.read_frame()
            
            if not ret or frame is None:
                print("영상이 끝났거나 읽기 실패.")
                break
            
            # 닭 탐지 (필요한 경우만)
            frame, chicken_count = self.detect_chickens(frame)
            
            # 상태 정보 추가
            frame = self._add_status_info(frame, chicken_count)
            
            # 크기 조절
            if self.scale_factor != 1.0:
                display_height, display_width = frame.shape[:2]
                new_width = int(display_width * self.scale_factor)
                new_height = int(display_height * self.scale_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 프레임 표시
            cv2.imshow('Chicken Tracking', frame)
            
            # 키 입력 대기
            wait_time = max(1, self.delay)  # 최소 1ms는 대기
            key = cv2.waitKey(wait_time) & 0xFF
            
            # 키 입력에 따른 동작
            if not self.handle_keypress(key):
                break
                
            # FPS 측정 및 출력
            processed_frames += 1
            if processed_frames % fps_update_interval == 0:
                frame_time_end = cv2.getTickCount()
                elapsed = (frame_time_end - frame_time_start) / cv2.getTickFrequency()
                current_fps = fps_update_interval / elapsed
                print(f"현재 처리 속도: {current_fps:.1f} FPS")
                frame_time_start = cv2.getTickCount()
        
        # 자원 해제
        self.cap.release()
        cv2.destroyAllWindows()