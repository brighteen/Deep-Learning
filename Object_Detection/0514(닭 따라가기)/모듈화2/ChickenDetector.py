import numpy as np
import time
from ultralytics import YOLO

class ChickenDetector:
    """YOLO 모델을 사용하여 닭을 감지하고 추적하는 클래스"""
    
    def __init__(self, model_path):
        """
        YOLO 모델을 로드합니다.
        
        Args:
            model_path (str): YOLO 모델 파일 경로
        """
        try:
            self.model = YOLO(model_path)
            self.enabled = True
            self.tracking_enabled = True  # 객체 추적 활성화 상태
            print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
            
            # 객체 추적을 위한 데이터 구조
            self.chicken_id_sets = []  # 같은 닭으로 판단된 ID의 집합들
            self.id_to_set_index = {}   # ID가 어느 집합에 속하는지 매핑
            self.id_to_position = {}    # ID별 위치 정보
            self.id_to_last_frame = {}  # ID별 마지막 등장 프레임
            self.id_to_last_time = {}   # ID별 마지막 등장 시간
            self.set_to_display_id = {} # 각 집합의 대표 ID
            
            self.frame_count = 0        # 현재 프레임 번호
            self.use_consistent_ids = False # 일관된 ID 사용 여부
            
            # 객체 추적 설정
            self.distance_threshold = 50  # 같은 객체로 판단할 최대 거리
            self.time_threshold = 2.0     # 같은 객체로 판단할 최대 시간 차이 (초)
            self.max_frames = 100         # 오래된 데이터 삭제 기준 프레임 수
        except Exception as e:
            print(f"YOLO 모델 로드 실패: {e}")
            self.enabled = False
            self.tracking_enabled = False
            self.model = None
    def detect(self, frame, conf_threshold=0.5):
        """
        주어진 프레임에서 닭을 탐지합니다.
        
        Args:
            frame: 탐지할 프레임
            conf_threshold: 탐지 확신도 임계값
            
        Returns:
            탐지 결과와 닭의 개수
        """
        if not self.enabled:
            return None, 0
        
        try:
            # 프레임 카운터 증가
            self.frame_count += 1
            current_time = time.time()
            
            if self.tracking_enabled:
                # 객체 추적 모드 - track() 메서드 사용
                # track을 사용하면 객체에 ID가 자동으로 할당됨
                results = self.model.track(frame, conf=conf_threshold, verbose=False, persist=True)[0]
                
                # ID가 할당된 결과만 유효하다고 봄
                if hasattr(results, 'boxes') and hasattr(results.boxes, 'id') and results.boxes.id is not None:
                    boxes = results.boxes
                    chicken_count = len(boxes)
                    
                    # 지속적 ID 추적이 활성화된 경우
                    if self.use_consistent_ids:
                        self._update_id_mapping(boxes, current_time)
                        
                        # ID 매핑 정보가 있으면 결과 수정
                        if hasattr(boxes, 'id') and boxes.id is not None:
                            results = self._modify_results_with_mapped_ids(results)
                else:
                    # 추적 실패 시 일반 탐지 결과 사용
                    results = self.model.predict(frame, conf=conf_threshold, verbose=False)[0]
                    boxes = results.boxes
                    chicken_count = len(boxes)
            else:
                # 일반 탐지 모드
                results = self.model.predict(frame, conf=conf_threshold, verbose=False)[0]
                boxes = results.boxes
                chicken_count = len(boxes)
            
            # 100프레임마다 오래된 데이터 정리
            if self.frame_count % 100 == 0:
                self._clean_up_stale_data()
                
            return results, chicken_count
        except Exception as e:
            print(f"객체 탐지/추적 중 오류 발생: {e}")
            # 오류 발생 시 빈 결과 반환
            results = self.model.predict(frame, conf=conf_threshold, verbose=False)[0]
            return results, 0
            
    def toggle_tracking(self):
        """객체 추적 모드를 켜고 끕니다."""
        if self.enabled:
            self.tracking_enabled = not self.tracking_enabled
            print(f"객체 추적: {'활성화' if self.tracking_enabled else '비활성화'}")
            return True
        return False
        
    def toggle_consistent_ids(self):
        """지속적인 ID 추적 기능을 켜고 끕니다."""
        if self.enabled and self.tracking_enabled:
            self.use_consistent_ids = not self.use_consistent_ids
            print(f"일관된 ID 추적: {'활성화' if self.use_consistent_ids else '비활성화'}")
            
            # 기존 추적 데이터 초기화
            if self.use_consistent_ids:
                self.chicken_id_sets = []
                self.id_to_set_index = {}
                self.id_to_position = {}
                self.id_to_last_frame = {}
                self.id_to_last_time = {}
                self.set_to_display_id = {}
            
            return True
        return False
        
    def _update_id_mapping(self, boxes, current_time):
        """
        탐지된 객체들의 ID 매핑을 업데이트합니다.
        
        Args:
            boxes: 탐지된 경계 상자들
            current_time: 현재 시간
        """
        if not hasattr(boxes, 'id') or boxes.id is None:
            return
            
        # 현재 탐지된 객체 IDs
        current_ids = boxes.id.cpu().numpy()
        
        # 중앙 위치 계산
        xyxy = boxes.xyxy.cpu().numpy()
        centers = []
        for box in xyxy:
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            centers.append((center_x, center_y))
        
        # 새롭게 등장한 ID 처리
        for i, chicken_id in enumerate(current_ids):
            chicken_id = int(chicken_id)
            center = centers[i]
            
            # 이미 알고 있는 ID인 경우, 위치 및 시간 정보만 업데이트
            if chicken_id in self.id_to_position:
                self.id_to_position[chicken_id] = center
                self.id_to_last_frame[chicken_id] = self.frame_count
                self.id_to_last_time[chicken_id] = current_time
                continue
                
            # 새로운 ID가 나타난 경우
            matched = False
            
            # 기존 객체 중 근처에 있었던 것이 있는지 확인
            for old_id, old_pos in self.id_to_position.items():
                # 거리 계산
                distance = np.sqrt((center[0] - old_pos[0])**2 + (center[1] - old_pos[1])**2)
                time_diff = current_time - self.id_to_last_time.get(old_id, 0)
                
                # 거리와 시간 모두 임계값 이내인 경우
                if distance < self.distance_threshold and time_diff < self.time_threshold:
                    # 같은 객체로 매칭
                    matched = True
                    
                    # 기존 ID가 속한 집합을 확인
                    if old_id in self.id_to_set_index:
                        set_index = self.id_to_set_index[old_id]
                        self.chicken_id_sets[set_index].add(chicken_id)
                        self.id_to_set_index[chicken_id] = set_index
                    else:
                        # 기존 ID가 집합에 없는 경우 - 새로운 집합 생성
                        self.chicken_id_sets.append({old_id, chicken_id})
                        set_index = len(self.chicken_id_sets) - 1
                        self.id_to_set_index[old_id] = set_index
                        self.id_to_set_index[chicken_id] = set_index
                        self.set_to_display_id[set_index] = min(chicken_id, old_id)  # 작은 ID를 대표로
                    break
                    
            # 매칭되지 않은 경우 새로운 객체로 등록
            if not matched:
                self.chicken_id_sets.append({chicken_id})
                set_index = len(self.chicken_id_sets) - 1
                self.id_to_set_index[chicken_id] = set_index
                self.set_to_display_id[set_index] = chicken_id
                
            # 위치 및 시간 정보 업데이트
            self.id_to_position[chicken_id] = center
            self.id_to_last_frame[chicken_id] = self.frame_count
            self.id_to_last_time[chicken_id] = current_time
            
    def _modify_results_with_mapped_ids(self, results):
        """
        탐지 결과의 ID를 매핑된 정보를 사용하여 수정합니다.
        """
        if not hasattr(results.boxes, 'id') or results.boxes.id is None:
            return results
            
        # 원본 ID 배열은 수정할 수 없으므로, 대신 렌더링 시 표시할 ID 정보를 저장
        current_ids = results.boxes.id.cpu().numpy()
        display_ids = []
        
        for i, chicken_id in enumerate(current_ids):
            chicken_id = int(chicken_id)
            # 일관된 ID 매핑 적용
            if chicken_id in self.id_to_set_index:
                set_index = self.id_to_set_index[chicken_id]
                display_id = self.set_to_display_id.get(set_index, chicken_id)
                display_ids.append(display_id)
            else:
                display_ids.append(chicken_id)
        
        # 표시 ID 정보를 결과 객체에 저장
        results.display_ids = display_ids
            
        return results
        
    def _clean_up_stale_data(self):
        """오래된 추적 데이터를 정리합니다."""
        # 오래된 ID 찾기
        stale_ids = []
        for chicken_id, last_frame in self.id_to_last_frame.items():
            if self.frame_count - last_frame > self.max_frames:
                stale_ids.append(chicken_id)
        
        # 오래된 데이터 삭제
        for chicken_id in stale_ids:
            set_index = self.id_to_set_index.pop(chicken_id, None)
            if set_index is not None and set_index < len(self.chicken_id_sets):
                self.chicken_id_sets[set_index].discard(chicken_id)
                
                # 집합이 비어있으면 대표 ID 매핑 삭제
                if not self.chicken_id_sets[set_index]:
                    self.set_to_display_id.pop(set_index, None)
                    
            self.id_to_position.pop(chicken_id, None)
            self.id_to_last_frame.pop(chicken_id, None)
            self.id_to_last_time.pop(chicken_id, None)
            
        # 빈 집합 제거 (인덱스 조정 필요)
        if len(self.chicken_id_sets) > 0:
            non_empty_indices = {}
            new_chicken_id_sets = []
            
            for i, id_set in enumerate(self.chicken_id_sets):
                if id_set:
                    non_empty_indices[i] = len(new_chicken_id_sets)
                    new_chicken_id_sets.append(id_set)
            
            # 인덱스 매핑 업데이트
            self.chicken_id_sets = new_chicken_id_sets
            
            # ID와 집합 인덱스 매핑 업데이트
            new_id_to_set_index = {}
            for chicken_id, old_index in self.id_to_set_index.items():
                if old_index in non_empty_indices:
                    new_id_to_set_index[chicken_id] = non_empty_indices[old_index]
            
            self.id_to_set_index = new_id_to_set_index
            
            # 대표 ID 매핑 업데이트
            new_set_to_display_id = {}
            for old_index, display_id in self.set_to_display_id.items():
                if old_index in non_empty_indices:
                    new_set_to_display_id[non_empty_indices[old_index]] = display_id
            
            self.set_to_display_id = new_set_to_display_id
