import cv2
import os
import numpy as np
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO  # YOLOv8 모델을 위한 라이브러리
from PIL import ImageFont, ImageDraw, Image
from collections import defaultdict

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

class ChickenTracker:
    """
    닭 ID 추적을 위한 클래스입니다.
    
    집합 기반 ID 시스템으로 유사한 ID를 그룹화하여 관리합니다.
    """
    def __init__(self, max_disappeared=30, iou_threshold=0.3):
        """
        ChickenTracker 클래스를 초기화합니다.
        
        Args:
            max_disappeared (int): 객체가 사라진 최대 프레임 수 (이후에 새 ID 할당)
            iou_threshold (float): 동일 객체로 간주할 IOU 임계값
        """
        self.id_sets = []         # ID 집합들의 리스트 (같은 닭으로 판단된 ID들을 하나의 집합으로 관리)
        self.disappeared = {}     # ID별 사라진 프레임 수
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.history = []         # 추적 데이터 (CSV 출력용)
        self.last_positions = {}  # ID별 마지막 위치 (x, y, w, h)
        self.last_results = None  # 프레임 스킵 시 이전 탐지 결과 저장
        self.last_time = 0.0      # 마지막 탐지 시간 저장
    
    def _calculate_iou(self, box1, box2):
        """
        두 박스의 IOU(Intersection Over Union)를 계산합니다.
        
        Args:
            box1: (x1, y1, x2, y2) 형태의 박스 좌표 (x1, y1: 좌상단, x2, y2: 우하단)
            box2: (x1, y1, x2, y2) 형태의 박스 좌표
            
        Returns:
            float: IOU 값 (0.0 ~ 1.0)
        """
        # 두 박스의 교차 영역 계산
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 교차 영역이 없는 경우
        if x2 < x1 or y2 < y1:
            return 0.0
        
        # 교차 영역 넓이
        intersection = (x2 - x1) * (y2 - y1)
        
        # 각 박스 넓이
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 합집합 넓이 = 두 박스 넓이의 합 - 교차 영역 넓이
        union = box1_area + box2_area - intersection
        
        # IOU 계산
        return intersection / union if union > 0 else 0.0
    
    def _get_set_id(self, chicken_id):
        """
        닭 ID가 속한 집합 인덱스를 반환합니다.
        
        Args:
            chicken_id: 닭 ID
            
        Returns:
            int: ID가 속한 집합 인덱스, 없으면 -1
        """
        for i, id_set in enumerate(self.id_sets):
            if chicken_id in id_set:
                return i
        return -1
    
    def _merge_sets(self, set_idx1, set_idx2):
        """
        두 ID 집합을 병합합니다.
        
        Args:
            set_idx1: 첫 번째 집합 인덱스
            set_idx2: 두 번째 집합 인덱스
        """
        if set_idx1 == set_idx2:
            return
            
        # 작은 인덱스의 집합에 큰 인덱스의 집합을 병합
        min_idx = min(set_idx1, set_idx2)
        max_idx = max(set_idx1, set_idx2)
        
        self.id_sets[min_idx] = self.id_sets[min_idx].union(self.id_sets[max_idx])
        del self.id_sets[max_idx]
    
    def update(self, frame_number, time_seconds, boxes, ids=None):
        """
        현재 프레임의 탐지 결과를 기반으로 추적 정보를 업데이트합니다.
        
        Args:
            frame_number: 현재 프레임 번호
            time_seconds: 현재 시간(초)
            boxes: 탐지된 객체의 바운딩 박스 리스트 [(x1, y1, x2, y2), ...]
            ids: 각 객체에 할당된 ID 리스트 (YOLOv8 track의 결과, 없으면 None)
            
        Returns:
            tuple: (업데이트된 ID 리스트, ID 집합 리스트)
        """
        # 현재 프레임에 탐지된 객체가 없는 경우
        if not boxes or len(boxes) == 0:
            # 모든 객체의 사라짐 카운트 증가
            for obj_id in list(self.last_positions.keys()):
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                
                # 사라짐 이벤트 기록
                self.history.append({
                    'frame': frame_number,
                    'time': time_seconds,
                    'chicken_id': obj_id,
                    'x1': None, 'y1': None, 'x2': None, 'y2': None,
                    'width': None, 'height': None,
                    'status': 'disappeared',
                    'set_id': self._get_set_id(obj_id),
                    'disappeared_frames': self.disappeared[obj_id]
                })
                
                # 최대 사라짐 프레임 수 초과 시 제거
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.last_positions[obj_id]
                    del self.disappeared[obj_id]
            
            return [], self.id_sets
        
        current_ids = ids if ids is not None else [-1] * len(boxes)
        
        # YOLOv8에 의한 ID 할당이 없는 경우, IOU 기반으로 이전 프레임의 객체와 매칭
        if ids is None:
            # 이전 프레임의 객체 위치가 있는 경우
            if self.last_positions:
                matched_indices = {}
                
                # 각 탐지 결과에 대해 가장 높은 IOU를 가진 이전 객체 찾기
                for i, box in enumerate(boxes):
                    best_iou = 0
                    best_id = None
                    
                    for obj_id, last_box in self.last_positions.items():
                        iou = self._calculate_iou(box, last_box)
                        if iou > best_iou and iou > self.iou_threshold:
                            best_iou = iou
                            best_id = obj_id
                    
                    if best_id is not None:
                        matched_indices[i] = best_id
                        # 사라짐 카운트 초기화
                        self.disappeared[best_id] = 0
                
                # 매칭된 객체 ID 할당
                for i, box in enumerate(boxes):
                    if i in matched_indices:
                        current_ids[i] = matched_indices[i]
                    else:
                        # 새 객체에 고유 ID 할당
                        new_id = max(list(self.last_positions.keys()) + [0]) + 1 if self.last_positions else 1
                        current_ids[i] = new_id
                        # 새 ID에 대한 집합 생성
                        self.id_sets.append({new_id})
            else:
                # 첫 프레임: 모든 객체에 새 ID 할당
                for i in range(len(boxes)):
                    current_ids[i] = i + 1
                    self.id_sets.append({i + 1})
        else:
            # YOLOv8이 할당한 ID 사용
            # 새로 나타난 ID 처리
            for id in current_ids:
                found = False
                for id_set in self.id_sets:
                    if id in id_set:
                        found = True
                        break
                
                if not found and id != -1:  # -1은 할당되지 않은 ID
                    self.id_sets.append({id})
        
        # 현재 프레임의 모든 객체 상태 업데이트
        for i, (box, obj_id) in enumerate(zip(boxes, current_ids)):
            # 박스 정보 확장 (x1, y1, x2, y2, width, height)
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # ID가 이전에 없던 새 ID인 경우
            if obj_id not in self.last_positions and obj_id != -1:
                # ID 집합에 추가
                is_in_set = False
                for id_set in self.id_sets:
                    if obj_id in id_set:
                        is_in_set = True
                        break
                
                if not is_in_set:
                    self.id_sets.append({obj_id})
            
            # 객체 상태 기록
            status = 'tracked'
            if obj_id in self.disappeared:
                if self.disappeared[obj_id] > 0:
                    status = 'reappeared'
                self.disappeared[obj_id] = 0
            
            # 위치 정보 업데이트
            self.last_positions[obj_id] = box
            
            # 객체 추적 정보 저장
            self.history.append({
                'frame': frame_number,
                'time': time_seconds,
                'chicken_id': obj_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': width, 'height': height,
                'status': status,
                'set_id': self._get_set_id(obj_id),
                'disappeared_frames': 0
            })
        
        # 이번 프레임에 나타나지 않은 객체 처리
        for obj_id in list(self.last_positions.keys()):
            if obj_id not in current_ids:
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                
                # 사라짐 이벤트 기록
                self.history.append({
                    'frame': frame_number,
                    'time': time_seconds,
                    'chicken_id': obj_id,
                    'x1': None, 'y1': None, 'x2': None, 'y2': None,
                    'width': None, 'height': None,
                    'status': 'disappeared',
                    'set_id': self._get_set_id(obj_id),
                    'disappeared_frames': self.disappeared[obj_id]
                })
                
                # 최대 사라짐 프레임 수 초과 시 제거
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.last_positions[obj_id]
                    del self.disappeared[obj_id]
        
        # ID 집합 업데이트: 박스 위치가 매우 가까운 경우 동일 닭으로 판단하여 집합 병합
        for i, (box1, id1) in enumerate(zip(boxes, current_ids)):
            for j, (box2, id2) in enumerate(zip(boxes, current_ids)):
                if i != j and id1 != id2:
                    if self._calculate_iou(box1, box2) > self.iou_threshold * 1.2:  # 높은 IOU 임계값
                        set_id1 = self._get_set_id(id1)
                        set_id2 = self._get_set_id(id2)
                        
                        if set_id1 != -1 and set_id2 != -1:
                            self._merge_sets(set_id1, set_id2)
        
        return current_ids, self.id_sets
    
    def get_tracking_stats(self):
        """
        추적 통계 정보를 계산합니다.
        
        Returns:
            dict: 추적 통계 정보
        """
        if not self.history:
            return {'id_sets': len(self.id_sets), 'total_unique_ids': 0, 'reappearance_events': 0}
        
        # 전체 고유 ID 수
        unique_ids = set()
        for item in self.history:
            if item['chicken_id'] != -1:
                unique_ids.add(item['chicken_id'])
        
        # 재등장 이벤트 수
        reappearance_events = sum(1 for item in self.history if item['status'] == 'reappeared')
        
        # ID 집합별 통계
        set_stats = {}
        for i, id_set in enumerate(self.id_sets):
            set_stats[f'set_{i}'] = {
                'ids': list(id_set),
                'count': len(id_set)
            }
        
        return {
            'id_sets': len(self.id_sets),
            'total_unique_ids': len(unique_ids),
            'reappearance_events': reappearance_events,
            'set_details': set_stats
        }
      def export_to_csv(self, filename=None):
        """
        닭 ID 변화 정보만 CSV 파일로 내보냅니다.
        
        Args:
            filename: 저장할 파일명 (None이면 자동 생성)
            
        Returns:
            str: 저장된 CSV 파일 경로
        """
        if not self.history:
            print("추적 데이터가 없습니다.")
            return None
        
        # 기본 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chicken_tracking_{timestamp}.csv"
        
        # ID 변화 데이터만 추출
        id_changes = []
        # 프레임별로 어떤 ID가 등장했는지 추적
        frame_ids = {}
        
        for item in self.history:
            frame = item['frame']
            chicken_id = item['chicken_id']
            status = item['status']
            set_id = item['set_id']
            
            # 프레임별로 ID 목록 정리
            if frame not in frame_ids:
                frame_ids[frame] = {}
            
            # ID가 유효한 경우만 기록
            if chicken_id != -1:
                frame_ids[frame][chicken_id] = {
                    'status': status,
                    'set_id': set_id
                }
        
        # 각 프레임에 어떤 ID들이 있었는지 기록
        for frame, ids in sorted(frame_ids.items()):
            time_seconds = frame / (fps if 'fps' in globals() else 30)  # fps가 없으면 기본값 30 사용
            
            id_changes.append({
                'frame': frame,
                'time': time_seconds,
                'active_ids': list(ids.keys()),
                'sets': [ids[id_key]['set_id'] for id_key in ids.keys()],
                'status': {id_key: data['status'] for id_key, data in ids.items()}
            })
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(id_changes)
        df.to_csv(filename, index=False)
        
        return filename


def track_chickens(model, frame, conf_threshold=0.5):
    """
    YOLO 모델로 닭을 추적합니다.
    
    Args:
        model: YOLO 모델
        frame: 처리할 프레임
        conf_threshold: 탐지 확신도 임계값
        
    Returns:
        results: 추적 결과
        chicken_count: 탐지된 닭의 수
    """
    # YOLOv8 모델의 track 메소드 사용
    results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)[0]
    
    # 결과에서 닭 개체 수 계산
    boxes = results.boxes
    chicken_count = len(boxes)
    
    return results, chicken_count


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
    try:
        model = YOLO(model_path)
        yolo_enabled = True
        print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        yolo_enabled = False
    
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
    
    # 비디오 정보 출력    print(f"원본 영상 크기: {width}x{height}")
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
    is_grid_mode = True  # 그리드 모드 여부
    selected_cell = target_cell  # 지정된 셀 선택
    is_grid_mode = False  # 확대 모드로 시작
    
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
                break        # 닭 탐지 및 추적 (YOLO가 활성화된 경우)
        chicken_count = 0
        chicken_ids = []
        chicken_sets = []
        should_detect = frame_count % frame_skip == 0  # frame_skip 간격으로 탐지 수행
        frame_count += 1
        
        if yolo_enabled and yolo_detection_active:
            if not is_grid_mode and selected_cell == target_cell:  # 지정된 셀에서만 닭 추적
                row, col = selected_cell
                cell_height = height // grid_size
                cell_width = width // grid_size
                x = col * cell_width
                y = row * cell_height
                cell_frame = frame[y:y+cell_height, x:x+cell_width]
                
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
                    plotted_cell = results.plot(labels=True, font_size=0.5, line_width=1)  # 기본 레이블
                    
                    # 추가: ID 집합 번호 표시
                    if results.boxes.id is not None:
                        boxes = results.boxes.xyxy.cpu().tolist()
                        ids = results.boxes.id.int().cpu().tolist()
                        for i, box in enumerate(boxes):
                            if i < len(ids):
                                x1, y1, x2, y2 = map(int, box)
                                chicken_id = ids[i]
                                set_id = tracker._get_set_id(chicken_id)
                                cv2.putText(plotted_cell, f"S{set_id}", (x1, y1-20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    # 결과를 클래스에 저장 (다음 프레임에서 사용하기 위해)
                    tracker.last_results = plotted_cell.copy()
                    tracker.last_time = time_pos
                    
                    # 결과를 프레임에 적용
                    frame[y:y+cell_height, x:x+cell_width] = plotted_cell
                else:
                    # 탐지를 건너뛰는 프레임에서는 이전 결과 사용
                    if hasattr(tracker, 'last_results') and tracker.last_results is not None:
                        # 이전 결과 표시
                        frame[y:y+cell_height, x:x+cell_width] = tracker.last_results
                        
                        # 현재 시간 정보 업데이트
                        if hasattr(tracker, 'last_time'):
                            display_time_skipped = put_text_on_image(
                                frame[y:y+cell_height, x:x+cell_width].copy(),
                                f"마지막 탐지: {tracker.last_time:.1f}초", 
                                (10, 30), 20, (0, 0, 255))
                            frame[y:y+cell_height, x:x+cell_width] = display_time_skipped
            
            elif is_grid_mode and should_detect:  # 그리드 모드이고 탐지 프레임에서만
                # 전체 프레임에서 닭 탐지 (추적 없음)
                results, chicken_count = detect_chickens(model, frame, conf_threshold)
                
                # 탐지 결과를 프레임에 그림 (레이블 크기와 선 두께 조정)
                frame = results.plot(labels=True, font_size=0.4, line_width=1)
        
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
                    
                    # 특별히 (2,1) 셀을 강조
                    if row == 2 and col == 1:
                        cv2.putText(display_frame, f"({row},{col})", (x, y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                        # 셀에 사각형 강조
                        cv2.rectangle(display_frame, 
                                    (col * cell_width, row * cell_height), 
                                    ((col+1) * cell_width, (row+1) * cell_height), 
                                    (0, 0, 255), 3)
                    else:
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
        display_frame = put_text_on_image(display_frame, f"프레임 스킵: {frame_skip}", (10, 330), 25, (0, 255, 0))
        
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
                
                # 추적 정보 표시
                if selected_cell == (2, 1) and not is_grid_mode:
                    stats = tracker.get_tracking_stats()
                    display_frame = put_text_on_image(display_frame, f"ID 집합: {stats['id_sets']}개", 
                                (10, 240), 25, (0, 255, 255))
                    display_frame = put_text_on_image(display_frame, f"총 고유 ID: {stats['total_unique_ids']}개", 
                                (10, 270), 25, (0, 255, 255))
                    display_frame = put_text_on_image(display_frame, f"재등장 이벤트: {stats['reappearance_events']}회", 
                                (10, 300), 25, (0, 255, 255))
        
        # 프레임 표시
        cv2.imshow('Video with Tracking', display_frame)
        
        # 키 입력 대기 (일시정지 상태일 때는 더 빠르게 반응하도록)
        wait_time = 1 if is_paused else delay
        key = cv2.waitKey(wait_time) & 0xFF
        
        # 키 입력에 따른 동작
        if key == ord('q'):  # 'q' 키: 종료
            print("사용자가 재생을 중지했습니다.")
            
            # 추적 데이터를 CSV로 저장
            if tracker.history:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = f"chicken_tracking_{timestamp}.csv"
                csv_file = tracker.export_to_csv(csv_path)
                print(f"추적 데이터를 {csv_file}에 저장했습니다.")
                
                # ID 집합 정보 출력
                stats = tracker.get_tracking_stats()
                print("\n=== 닭 추적 통계 ===")
                print(f"총 ID 집합 수: {stats['id_sets']}개")
                print(f"총 고유 ID 수: {stats['total_unique_ids']}개")
                print(f"재등장 이벤트 수: {stats['reappearance_events']}회")
                print("\n=== ID 집합별 세부 정보 ===")
                for set_name, set_info in stats['set_details'].items():
                    print(f"{set_name}: {set_info['ids']} (총 {set_info['count']}개)")
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
            prompt_frame = display_frame.copy()
            prompt_frame = put_text_on_image(prompt_frame, 
                                             f"현재 프레임 스킵: {frame_skip}, + 키로 증가, - 키로 감소", 
                                             (10, 400), 30, (255, 0, 0))
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
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

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

if __name__ == "__main__":
    import argparse
    
    # 명령행 인자 파서 생성
    parser = argparse.ArgumentParser(description="닭 ID 추적 프로그램")
    parser.add_argument("--video", default=r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20221105100432.mp4", 
                        help="영상 파일 경로")
    parser.add_argument("--model", default=r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt", 
                        help="YOLO 모델 경로")
    parser.add_argument("--grid_size", type=int, default=5, help="그리드 크기 (NxN)")
    parser.add_argument("--scale", type=float, default=0.7, help="영상 크기 조절 비율")
    parser.add_argument("--max_time", type=int, default=60, help="최대 재생 시간(초)")
    parser.add_argument("--frame_skip", type=int, default=5, help="프레임 건너뛰기 (높을수록 빠름)")
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
