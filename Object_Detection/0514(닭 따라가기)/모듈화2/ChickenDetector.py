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
            self.distance_threshold = 25  # 같은 객체로 판단할 최대 거리 (줄임: 50->25)
            self.time_threshold = 1.5     # 같은 객체로 판단할 최대 시간 차이 (초) (줄임: 2.0->1.5)
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
        current_id_set = set([int(id) for id in current_ids])
        
        # 중앙 위치 계산
        xyxy = boxes.xyxy.cpu().numpy()
        centers = []
        for box in xyxy:
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            centers.append((center_x, center_y))
        
        # 현재 프레임에서 매칭을 추적하기 위한 변수
        already_matched_old_ids = set()
        already_matched_new_ids = set()
        
        # 최근에 사라진 ID 식별 (현재 프레임에서 보이지 않는 ID 중, 시간 임계값 내에 있는 ID)
        recently_disappeared_ids = {}
        for old_id, last_time in self.id_to_last_time.items():
            if old_id not in current_id_set:
                time_diff = current_time - last_time
                if time_diff < self.time_threshold:
                    recently_disappeared_ids[old_id] = (self.id_to_position[old_id], time_diff)
        
        # 각 새로운 ID와 최근에 사라진 ID 사이의 거리를 계산하여 매칭
        potential_matches = []
        for i, chicken_id in enumerate(current_ids):
            chicken_id = int(chicken_id)
            center = centers[i]
            
            # 이미 알고 있는 ID인 경우, 위치 및 시간 정보만 업데이트
            if chicken_id in self.id_to_position:
                self.id_to_position[chicken_id] = center
                self.id_to_last_frame[chicken_id] = self.frame_count
                self.id_to_last_time[chicken_id] = current_time
                already_matched_new_ids.add(chicken_id)
                continue
                
            # 새로운 ID가 나타난 경우, 최근에 사라진 ID와의 거리만 계산 (성능 최적화)
            for old_id, (old_pos, time_diff) in recently_disappeared_ids.items():
                # 이미 매칭된 기존 ID는 건너뜀
                if old_id in already_matched_old_ids:
                    continue
                    
                # 거리 계산 (Euclidean 대신 제곱 거리를 사용하여 sqrt 연산 제거 - 추가 최적화)
                distance_squared = (center[0] - old_pos[0])**2 + (center[1] - old_pos[1])**2
                
                # 거리 제곱과 시간 조건 확인 (임계값도 제곱해서 비교)
                if distance_squared < (self.distance_threshold**2):
                    # 거리와 시간을 가중치로 조합한 점수 계산 (가까울수록, 시간차가 적을수록 점수 높음)
                    match_score = distance_squared + (time_diff * 10)  # 시간 가중치 추가
                    potential_matches.append((match_score, i, chicken_id, old_id))
        
        # 매칭 점수가 낮은 순서로 정렬 (낮을수록 더 좋은 매치)
        potential_matches.sort()
        for score, i, chicken_id, old_id in potential_matches:
            # 이미 매칭된 ID는 건너뜀
            if old_id in already_matched_old_ids or chicken_id in already_matched_new_ids:
                continue
                  
            # 매칭 성공
            already_matched_old_ids.add(old_id)
            already_matched_new_ids.add(chicken_id)
            center = centers[i]
            
            # 기존 ID가 속한 집합을 확인 (해시 테이블 사용으로 O(1) 접근)
            if old_id in self.id_to_set_index:
                set_index = self.id_to_set_index[old_id]
                self.chicken_id_sets[set_index].add(chicken_id)
                self.id_to_set_index[chicken_id] = set_index
            else:
                # 기존 ID가 집합에 없는 경우 - 새로운 집합 생성
                set_index = len(self.chicken_id_sets)
                self.chicken_id_sets.append({old_id, chicken_id})
                self.id_to_set_index[old_id] = set_index
                self.id_to_set_index[chicken_id] = set_index
                self.set_to_display_id[set_index] = min(chicken_id, old_id)  # 작은 ID를 대표로
                
            # 위치 및 시간 정보 업데이트
            self.id_to_position[chicken_id] = center
            self.id_to_last_frame[chicken_id] = self.frame_count
            self.id_to_last_time[chicken_id] = current_time
        
        # 매칭되지 않은 새 ID들을 병합 처리 (한 번에 처리하여 루프 최적화)
        unmatched_ids = [int(id) for i, id in enumerate(current_ids) if int(id) not in already_matched_new_ids]
        
        # 매칭되지 않은 새로운 ID들을 한 번에 처리 (배치 처리)
        if unmatched_ids:
            # 배치 처리를 위한 데이터 준비
            batch_sets = []
            batch_indices = []
            batch_positions = {}
            
            for i, chicken_id in enumerate(current_ids):
                chicken_id = int(chicken_id)
                if chicken_id not in already_matched_new_ids:
                    # 새 집합 인덱스 생성
                    set_index = len(self.chicken_id_sets) + len(batch_sets)
                    
                    # 임시 배치 데이터에 추가
                    batch_sets.append({chicken_id})
                    batch_indices.append((chicken_id, set_index))
                    batch_positions[chicken_id] = centers[i]
            
            # 배치 데이터를 한 번에 추가
            self.chicken_id_sets.extend(batch_sets)
            
            # 인덱스 매핑 업데이트
            for chicken_id, set_index in batch_indices:
                self.id_to_set_index[chicken_id] = set_index
                self.set_to_display_id[set_index] = chicken_id
                
                # 위치 및 시간 정보 업데이트
                self.id_to_position[chicken_id] = batch_positions[chicken_id]
                self.id_to_last_frame[chicken_id] = self.frame_count
                self.id_to_last_time[chicken_id] = current_time
                
    def _modify_results_with_mapped_ids(self, results):
        """
        탐지 결과의 ID를 매핑된 정보를 사용하여 수정합니다.
        """
        if not hasattr(results.boxes, 'id') or results.boxes.id is None:
            return results
            
        # 캐싱을 위한 변수 (지역 캐시 사용으로 속도 향상)
        id_to_set_index_cache = self.id_to_set_index
        chicken_id_sets_cache = self.chicken_id_sets
        set_to_display_id_cache = self.set_to_display_id
        
        # 원본 ID 배열은 수정할 수 없으므로, 대신 렌더링 시 표시할 ID 정보를 저장
        current_ids = results.boxes.id.cpu().numpy()
        display_ids = []
        id_sets = []  # 각 객체가 속한 ID 집합
        
        # 빠른 조회를 위해 사전 계산된 캐시 사용
        for chicken_id in current_ids:
            chicken_id = int(chicken_id)
            # 일관된 ID 매핑 적용 (캐시된 데이터 사용)
            if chicken_id in id_to_set_index_cache:
                set_index = id_to_set_index_cache[chicken_id]
                if set_index < len(chicken_id_sets_cache):  # 유효 범위 확인
                    display_id = set_to_display_id_cache.get(set_index, chicken_id)
                    display_ids.append(display_id)
                    id_sets.append(chicken_id_sets_cache[set_index])
                else:
                    # 인덱스가 유효하지 않은 경우 기본값
                    display_ids.append(chicken_id)
                    id_sets.append({chicken_id})
            else:
                display_ids.append(chicken_id)
                id_sets.append({chicken_id})
        
        # 표시 ID 정보 및 ID 집합 정보를 결과 객체에 저장
        results.display_ids = display_ids
        results.id_sets = id_sets
            
        return results
            
    def _clean_up_stale_data(self):
        """오래된 추적 데이터를 정리합니다."""
        # 성능 최적화: 오래된 ID를 한 번의 순회로 식별
        frame_threshold = self.frame_count - self.max_frames
        stale_ids = [chicken_id for chicken_id, last_frame in self.id_to_last_frame.items() 
                    if last_frame <= frame_threshold]
        
        # 성능 최적화: 데이터가 없으면 빠른 반환 (불필요한 처리 방지)
        if not stale_ids:
            return
        
        # 집합 수정 필요한지 추적
        affected_sets = set()
        
        # 오래된 데이터 삭제 (배치 처리)
        for chicken_id in stale_ids:
            set_index = self.id_to_set_index.pop(chicken_id, None)
            if set_index is not None and set_index < len(self.chicken_id_sets):
                self.chicken_id_sets[set_index].discard(chicken_id)
                # 집합이 변경되었음을 기록
                affected_sets.add(set_index)
                
            # 한 번에 삭제 (맵 조회 최소화)
            self.id_to_position.pop(chicken_id, None)
            self.id_to_last_frame.pop(chicken_id, None)
            self.id_to_last_time.pop(chicken_id, None)
        
        # 빈 집합 식별 (최적화: 영향받은 집합만 확인)
        empty_sets = [idx for idx in affected_sets if not self.chicken_id_sets[idx]]
        
        # 빈 집합의 대표 ID 매핑 삭제 (성능 최적화: 한 번의 루프로 처리)
        for set_index in empty_sets:
            self.set_to_display_id.pop(set_index, None)
            
        # 빈 집합 제거 (인덱스 조정 필요)
        # 최적화: 빈 집합이 있을 때만 처리
        if empty_sets and self.chicken_id_sets:
            # 새로운 집합 구조 생성
            non_empty_indices = {}
            new_chicken_id_sets = []
            
            # 성능 최적화: 한 번의 순회로 빈 집합 필터링 및 인덱스 매핑 
            for i, id_set in enumerate(self.chicken_id_sets):
                if id_set:  # 비어있지 않은 집합만 유지
                    non_empty_indices[i] = len(new_chicken_id_sets)
                    new_chicken_id_sets.append(id_set)
            
            # 전체 데이터 구조 업데이트 (성능 최적화: 한 번에 업데이트)
            # 새로운 집합 배열 적용
            self.chicken_id_sets = new_chicken_id_sets
            
            # ID와 집합 인덱스 매핑 업데이트 (딕셔너리 컴프리헨션으로 최적화)
            self.id_to_set_index = {
                chicken_id: non_empty_indices[old_index] 
                for chicken_id, old_index in self.id_to_set_index.items() 
                if old_index in non_empty_indices
            }
            
            # 대표 ID 매핑 업데이트 (딕셔너리 컴프리헨션으로 최적화)
            self.set_to_display_id = {
                non_empty_indices[old_index]: display_id 
                for old_index, display_id in self.set_to_display_id.items() 
                if old_index in non_empty_indices
            }
