import cv2
import os
import numpy as np
import time
from ultralytics import YOLO
import psutil  # 메모리 사용량 모니터링용

class TrackedObject:
    """객체 추적 정보를 효율적으로 저장하는 클래스"""
    
    def __init__(self, obj_id, box, frame_count, timestamp):
        """추적 객체 초기화"""
        self.id = obj_id
        self.box = box
        self.last_frame = frame_count
        self.last_time = timestamp
        self.size = (box[2] - box[0]) * (box[3] - box[1])
        self.center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self.histogram = None
        self.speed = 0.0
        self.set_index = None  # 객체가 속한 ID 집합의 인덱스
        self.centers_history = [self.center]  # 중심점 이동 경로
        self.max_trail_length = 30  # 경로 추적을 위한 최대 포인트 수

    def update(self, box, frame_count, timestamp, frame=None):
        """객체 정보 업데이트"""
        prev_center = self.center
        
        # 박스 및 기본 정보 업데이트
        self.box = box
        self.last_frame = frame_count
        self.last_time = timestamp
        self.size = (box[2] - box[0]) * (box[3] - box[1])
        self.center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        
        # 중심점 기록 업데이트
        self.centers_history.append(self.center)
        if len(self.centers_history) > self.max_trail_length:
            self.centers_history.pop(0)
        
        # 이동 속도 계산
        moved_distance = ((self.center[0] - prev_center[0]) ** 2 + 
                          (self.center[1] - prev_center[1]) ** 2) ** 0.5
        self.speed = moved_distance
        
        # 히스토그램 계산 (프레임이 제공된 경우에만)
        if frame is not None and frame_count % 10 == 0:
            self.histogram = self.calculate_histogram(frame, box)
            
        return self

    def calculate_histogram(self, frame, box):
        """객체 영역의 히스토그램 계산"""
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # 바운딩 박스가 프레임 경계를 벗어나지 않도록 조정
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        
        # 너무 작은 영역은 히스토그램 계산에서 오류가 발생할 수 있으므로 체크
        if x2 <= x1 or y2 <= y1:
            return None
        
        # 바운딩 박스 영역 추출
        roi = frame[y1:y2, x1:x2]
        
        try:
            # HSV 색상 공간으로 변환 (색상 비교에 더 효과적)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 히스토그램 계산 (H와 S 채널만 사용)
            hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
            
            # 히스토그램 정규화
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            return hist
        except:
            return None


class IDManager:
    """객체 ID를 효율적으로 관리하는 클래스"""
    
    def __init__(self, config=None):
        """ID 관리자 초기화"""
        # 기본 설정값 정의
        default_config = {
            'base_distance_threshold': 10,
            'iou_threshold': 0.2,
            'size_difference_threshold': 1.5,
            'histogram_similarity_threshold': 0.6,
            'max_frames': 100,
            'max_trail_length': 30
        }
        
        # 사용자 설정과 기본 설정 병합
        self.config = default_config
        if config:
            self.config.update(config)
            
        # 객체 추적 데이터
        self.tracked_objects = {}  # id -> TrackedObject
        self.chicken_id_sets = []  # 같은 닭으로 판단된 ID의 집합들
        self.id_to_set_index = {}  # ID가 어느 집합에 속하는지 매핑
        self.set_to_display_id = {} # 각 집합의 대표 ID
        
        # ID 관계 추적 데이터
        self.id_transitions = {}   # ID 전환 기록 (이전 ID -> 새 ID)
        self.id_relation_confidence = {}  # (id1, id2) 형태의 키에 신뢰도 점수 저장
        
        # 상태 정보
        self.frame_count = 0  # 현재 프레임 번호
        self.scene_complexity = 0.5  # 장면 복잡도 (0.0~1.0)
        self.avg_object_speed = 0.0  # 평균 객체 이동 속도
        
        # 적응형 임계값
        self.current_distance_threshold = self.config['base_distance_threshold']
        
        # 메모리 모니터링
        self.last_memory_check = 0
        self.memory_check_interval = 5  # 5초마다 메모리 확인
        
        # 디버그 정보
        self.last_debug_time = time.time()
        self.debug_interval = 30  # 30초마다 디버그 정보 출력
        self.verbose = False
    
    def update(self, boxes_with_ids, frame, current_time):
        """
        탐지된 객체들의 ID 정보를 업데이트합니다
        
        Args:
            boxes_with_ids: [(x1, y1, x2, y2, id), ...] 형태의 박스 및 ID 리스트
            frame: 현재 프레임 이미지
            current_time: 현재 시간
            
        Returns:
            업데이트된 ID 정보와 함께 튜플 (boxes_with_display_ids, changes)
        """
        # 프레임 카운터 증가
        self.frame_count += 1
        
        # 현재 메모리 사용량 확인 (5초마다)
        if time.time() - self.last_memory_check > self.memory_check_interval:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            if self.verbose:
                print(f"현재 메모리 사용량: {mem_info.rss / 1024 / 1024:.1f} MB")
            self.last_memory_check = time.time()
        
        # 현재 박스와 ID 정보 추출
        detected_boxes = []
        detected_ids = []
        
        for box_with_id in boxes_with_ids:
            box = (box_with_id[0], box_with_id[1], box_with_id[2], box_with_id[3])
            obj_id = int(box_with_id[4]) if len(box_with_id) > 4 else -1
            detected_boxes.append(box)
            detected_ids.append(obj_id)
        
        # 장면 복잡도 업데이트 (매 10프레임마다)
        if self.frame_count % 10 == 0:
            self.scene_complexity = self._calculate_scene_complexity(detected_boxes, frame.shape)
            
            # 모든 객체의 평균 속도 계산
            speeds = [obj.speed for obj in self.tracked_objects.values() if obj.speed > 0]
            self.avg_object_speed = sum(speeds) / len(speeds) if speeds else 0.0
            
            # 적응형 거리 임계값 업데이트
            overall_confidence = 0.5  # 기본 신뢰도
            self.current_distance_threshold = self._update_adaptive_threshold(
                self.config['base_distance_threshold'], 
                self.scene_complexity, 
                self.avg_object_speed, 
                overall_confidence
            )
        
        # 현재 시점에서의 객체 ID 업데이트
        changes = []  # ID 변경 로그
        
        for i, (box, obj_id) in enumerate(zip(detected_boxes, detected_ids)):
            # ID가 없으면 가장 가까운 객체의 ID로 복구 시도
            if obj_id == -1:
                best_match_id = None
                best_match_score = float('inf')
                best_match_box = None
                
                # 이전에 추적하던 모든 객체와 비교
                for tracked_id, tracked_obj in self.tracked_objects.items():
                    # 최근 5프레임 이내에 본 객체만 고려 (성능 최적화)
                    if self.frame_count - tracked_obj.last_frame > 5:
                        continue
                    
                    # IoU 계산
                    iou = self._calculate_iou(box, tracked_obj.box)
                    if iou > 0.5:
                        # IoU가 높으면 해당 ID 사용
                        if best_match_score > 1.0 - iou:
                            best_match_score = 1.0 - iou
                            best_match_id = tracked_id
                            best_match_box = tracked_obj.box
                
                # 매칭된 ID가 있으면 사용
                if best_match_id is not None:
                    obj_id = best_match_id
                    detected_ids[i] = obj_id
                    changes.append(f"ID 복구: {obj_id} (IoU={1.0-best_match_score:.2f})")
            
            # 객체 정보 업데이트 또는 새로 추가
            if obj_id != -1:
                # 기존 객체 업데이트
                if obj_id in self.tracked_objects:
                    self.tracked_objects[obj_id].update(box, self.frame_count, current_time, frame)
                else:
                    # 새 객체 추가
                    self.tracked_objects[obj_id] = TrackedObject(obj_id, box, self.frame_count, current_time)
                    
                    # 새 객체의 히스토그램 계산
                    self.tracked_objects[obj_id].histogram = self.tracked_objects[obj_id].calculate_histogram(frame, box)
                    
                    # 새 ID 집합 생성
                    set_index = len(self.chicken_id_sets)
                    self.chicken_id_sets.append({obj_id})
                    self.id_to_set_index[obj_id] = set_index
                    self.set_to_display_id[set_index] = obj_id
                    
                    # 새 객체와 기존 객체 간의 관계 분석
                    self._analyze_new_object_relations(obj_id, self.tracked_objects[obj_id])
        
        # ID 집합 병합 수행
        self._merge_id_sets(detected_boxes, detected_ids)
        
        # 오래된 데이터 정리
        if self.frame_count % self.config['max_frames'] == 0:
            self._clean_up_stale_data()
        
        # 디버그 정보 출력
        if self.verbose and time.time() - self.last_debug_time > self.debug_interval:
            self._print_debug_info()
            self.last_debug_time = time.time()
        
        # 최종 결과 (디스플레이 ID) 생성
        boxes_with_display_ids = []
        for i, (box, obj_id) in enumerate(zip(detected_boxes, detected_ids)):
            if obj_id != -1:
                display_id = self.get_display_id(obj_id)
                boxes_with_display_ids.append((*box, obj_id, display_id))
            else:
                boxes_with_display_ids.append((*box, -1, -1))
        
        return boxes_with_display_ids, changes
    
    def get_display_id(self, obj_id):
        """객체의 대표 ID 반환"""
        if obj_id in self.id_to_set_index:
            set_index = self.id_to_set_index[obj_id]
            if set_index < len(self.chicken_id_sets):
                return self.set_to_display_id.get(set_index, obj_id)
        return obj_id
    
    def get_id_set(self, obj_id):
        """객체가 속한 ID 집합 반환"""
        if obj_id in self.id_to_set_index:
            set_index = self.id_to_set_index[obj_id]
            if set_index < len(self.chicken_id_sets):
                return self.chicken_id_sets[set_index]
        return {obj_id}
    
    def get_object_trail(self, obj_id):
        """객체의 이동 경로 반환"""
        # 객체의 실제 ID가 대표 ID로 매핑되어 있으면 대표 ID의 경로 사용
        display_id = self.get_display_id(obj_id)
        
        if display_id in self.tracked_objects:
            return self.tracked_objects[display_id].centers_history
        elif obj_id in self.tracked_objects:
            return self.tracked_objects[obj_id].centers_history
        
        return []
    
    def save_id_sets_to_file(self, filename):
        """ID 집합을 파일로 저장"""
        final_id_sets = self._get_final_id_sets()
        
        try:
            with open(filename, 'w') as f:
                # 대표 ID를 기준으로 정렬
                for display_id in sorted(final_id_sets.keys()):
                    id_set = final_id_sets[display_id]
                    if len(id_set) > 1:  # 하나 이상의 ID가 있는 집합만 저장
                        id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                        f.write(f"{display_id}: {id_set_str}\n")
            return True
        except Exception as e:
            print(f"파일 저장 오류: {e}")
            return False
    
    def _analyze_new_object_relations(self, new_id, new_obj):
        """새 객체와 기존 객체들 간의 관계 분석"""
        for old_id, old_obj in self.tracked_objects.items():
            # 자기 자신은 제외
            if old_id == new_id:
                continue
                
            # 최근에 본 객체만 비교 (30프레임 이내)
            if self.frame_count - old_obj.last_frame > 30:
                continue
            
            # 거리 계산
            distance = ((new_obj.center[0] - old_obj.center[0])**2 + 
                       (new_obj.center[1] - old_obj.center[1])**2) ** 0.5
            
            # 너무 멀리 있는 객체는 제외
            if distance > self.current_distance_threshold * 3:
                continue
            
            # 히스토그램 유사도 계산
            hist_similarity = 0.0
            if new_obj.histogram is not None and old_obj.histogram is not None:
                hist_similarity = self._compare_histograms(new_obj.histogram, old_obj.histogram)
            
            # 신뢰도 점수 계산
            distance_confidence = max(0, 1.0 - (distance / (self.current_distance_threshold * 3)))
            confidence_score = distance_confidence * 0.5 + hist_similarity * 0.5
            
            # 신뢰도 점수가 0.3 이상인 경우만 저장
            if confidence_score >= 0.3:
                # ID 관계 등록
                id_pair = tuple(sorted([old_id, new_id]))
                self.id_relation_confidence[id_pair] = confidence_score
                
                # ID 전환 관계 기록
                if new_id not in self.id_transitions:
                    self.id_transitions[new_id] = []
                if old_id not in self.id_transitions:
                    self.id_transitions[old_id] = []
                    
                # 상호 전환 관계 기록
                if old_id not in self.id_transitions[new_id]:
                    self.id_transitions[new_id].append(old_id)
                if new_id not in self.id_transitions[old_id]:
                    self.id_transitions[old_id].append(new_id)
    
    def _merge_id_sets(self, boxes, ids):
        """ID 집합 병합 로직"""
        # 각 객체 쌍에 대해 ID 집합 병합 검사
        for i in range(len(boxes)):
            id1 = ids[i]
            if id1 == -1:
                continue
            
            box1 = boxes[i]
            obj1 = self.tracked_objects.get(id1)
            if obj1 is None:
                continue
                
            for j in range(i + 1, len(boxes)):
                id2 = ids[j]
                if id2 == -1:
                    continue
                
                # 이미 같은 집합이면 건너뜀
                if (id1 in self.id_to_set_index and id2 in self.id_to_set_index and 
                    self.id_to_set_index[id1] == self.id_to_set_index[id2]):
                    continue
                
                box2 = boxes[j]
                obj2 = self.tracked_objects.get(id2)
                if obj2 is None:
                    continue
                
                # 병합 점수 계산
                merge_score = self._calculate_merge_score(obj1, obj2)
                
                # 병합 여부 결정
                if merge_score >= 0.55:
                    # 신뢰도 점수 업데이트
                    id_pair = tuple(sorted([id1, id2]))
                    self.id_relation_confidence[id_pair] = max(
                        self.id_relation_confidence.get(id_pair, 0.0),
                        merge_score
                    )
                    
                    # 두 ID 집합 병합
                    self._merge_two_sets(id1, id2)
    
    def _calculate_merge_score(self, obj1, obj2):
        """두 객체 간의 병합 점수 계산"""
        # 거리 계산
        distance = ((obj1.center[0] - obj2.center[0])**2 + 
                   (obj1.center[1] - obj2.center[1])**2) ** 0.5
        
        # 거리 점수 (가까울수록 높음)
        distance_score = max(0.0, 1.0 - (distance / self.current_distance_threshold))
        
        # 크기 비율
        size_ratio = max(obj1.size, obj2.size) / min(obj1.size, obj2.size) if min(obj1.size, obj2.size) > 0 else float('inf')
        
        # 크기 유사성 점수
        size_score = max(0.0, 1.0 - ((size_ratio - 1.0) / self.config['size_difference_threshold']))
        
        # IoU 계산
        iou = self._calculate_iou(obj1.box, obj2.box)
        
        # 히스토그램 유사도 계산
        hist_similarity = 0.0
        if obj1.histogram is not None and obj2.histogram is not None:
            hist_similarity = self._compare_histograms(obj1.histogram, obj2.histogram)
        
        # 움직임 일관성 확인
        movement_consistent = self._is_consistent_movement(obj1, obj2)
        movement_score = 1.0 if movement_consistent else 0.0
        
        # 기존 신뢰도 가져오기
        id_pair = tuple(sorted([obj1.id, obj2.id]))
        confidence_score = self.id_relation_confidence.get(id_pair, 0.0)
        
        # 종합 점수 계산
        merge_score = (
            distance_score * 0.4 + 
            size_score * 0.15 + 
            hist_similarity * 0.2 + 
            movement_score * 0.15 + 
            confidence_score * 0.1
        )
        
        return merge_score
    
    def _merge_two_sets(self, id1, id2):
        """두 ID 집합 병합"""
        # 각 ID가 속한 집합 확인
        if id1 not in self.id_to_set_index or id2 not in self.id_to_set_index:
            return False
            
        set_index1 = self.id_to_set_index[id1]
        set_index2 = self.id_to_set_index[id2]
        
        # 이미 같은 집합이면 건너뜀
        if set_index1 == set_index2:
            return False
        
        # 작은 인덱스 집합으로 병합
        from_index = max(set_index1, set_index2)
        to_index = min(set_index1, set_index2)
        
        # ID 전환 관계 기록
        for from_id in self.chicken_id_sets[from_index]:
            for to_id in self.chicken_id_sets[to_index]:
                if from_id != to_id:
                    # 양방향 ID 전환 기록
                    if from_id not in self.id_transitions:
                        self.id_transitions[from_id] = []
                    if to_id not in self.id_transitions:
                        self.id_transitions[to_id] = []
                    
                    # 중복 방지
                    if to_id not in self.id_transitions[from_id]:
                        self.id_transitions[from_id].append(to_id)
                    if from_id not in self.id_transitions[to_id]:
                        self.id_transitions[to_id].append(from_id)
        
        # 병합 수행
        from_set = self.chicken_id_sets[from_index]
        self.chicken_id_sets[to_index].update(from_set)
        
        # ID 매핑 업데이트
        for moved_id in from_set:
            self.id_to_set_index[moved_id] = to_index
        
        # 대표 ID 업데이트
        self.set_to_display_id[to_index] = min(
            self.set_to_display_id.get(to_index, float('inf')),
            self.set_to_display_id.get(from_index, float('inf'))
        )
        
        # 병합된 집합 비우기
        self.chicken_id_sets[from_index] = set()
        self.set_to_display_id.pop(from_index, None)
        
        return True
    
    def _clean_up_stale_data(self):
        """오래된 데이터 정리"""
        # 프레임 임계값 계산
        frame_threshold = self.frame_count - self.config['max_frames']
        
        # 오래된 ID 식별
        stale_ids = [
            obj_id for obj_id, obj in self.tracked_objects.items() 
            if obj.last_frame <= frame_threshold
        ]
        
        # 데이터가 없으면 빠른 반환
        if not stale_ids:
            return
        
        # 영향 받는 집합 추적
        affected_sets = set()
        
        # 오래된 데이터 삭제
        for obj_id in stale_ids:
            # ID 집합 인덱스 확인
            set_index = self.id_to_set_index.pop(obj_id, None)
            
            if set_index is not None and set_index < len(self.chicken_id_sets):
                self.chicken_id_sets[set_index].discard(obj_id)
                affected_sets.add(set_index)
            
            # 객체 데이터 삭제
            self.tracked_objects.pop(obj_id, None)
            
            # ID 관계 데이터 정리
            if obj_id in self.id_transitions:
                self.id_transitions.pop(obj_id)
            
            # 모든 전환 관계에서 해당 ID 제거
            for other_id, transitions in self.id_transitions.items():
                if obj_id in transitions:
                    transitions.remove(obj_id)
        
        # ID 관계 신뢰도 정리
        stale_pairs = []
        for id_pair in self.id_relation_confidence.keys():
            if id_pair[0] in stale_ids or id_pair[1] in stale_ids:
                stale_pairs.append(id_pair)
        
        for pair in stale_pairs:
            self.id_relation_confidence.pop(pair, None)
        
        # 빈 집합 처리
        empty_sets = [idx for idx in affected_sets if not self.chicken_id_sets[idx]]
        
        # 빈 집합의 대표 ID 삭제
        for idx in empty_sets:
            self.set_to_display_id.pop(idx, None)
        
        # 집합 구조 재정렬 (빈 집합 제거 후 인덱스 재조정)
        if empty_sets:
            self._reindex_id_sets()
    
    def _reindex_id_sets(self):
        """ID 집합 인덱스 재조정"""
        # 비어있지 않은 집합만 유지
        new_sets = [s for s in self.chicken_id_sets if s]
        
        if not new_sets:
            self.chicken_id_sets = []
            self.id_to_set_index = {}
            self.set_to_display_id = {}
            return
        
        # 새로운 인덱스 매핑 생성
        old_to_new = {}
        for new_idx, old_idx in enumerate([i for i, s in enumerate(self.chicken_id_sets) if s]):
            old_to_new[old_idx] = new_idx
        
        # ID 집합 업데이트
        self.chicken_id_sets = new_sets
        
        # ID -> 집합 인덱스 매핑 업데이트
        new_id_to_set_index = {}
        for obj_id, old_idx in self.id_to_set_index.items():
            if old_idx in old_to_new:
                new_id_to_set_index[obj_id] = old_to_new[old_idx]
        
        self.id_to_set_index = new_id_to_set_index
        
        # 대표 ID 매핑 업데이트
        new_set_to_display_id = {}
        for old_idx, display_id in self.set_to_display_id.items():
            if old_idx in old_to_new:
                new_set_to_display_id[old_to_new[old_idx]] = display_id
        
        self.set_to_display_id = new_set_to_display_id
    
    def _get_final_id_sets(self):
        """최종 ID 집합 획득"""
        # ID 집합을 대표 ID -> 집합원 형태로 변환
        final_sets = {}
        
        # 1. 먼저 모든 ID 집합을 대표 ID 기준으로 정리
        for set_idx, id_set in enumerate(self.chicken_id_sets):
            if not id_set:
                continue
                
            display_id = self.set_to_display_id.get(set_idx, min(id_set))
            
            if display_id in final_sets:
                final_sets[display_id].update(id_set)
            else:
                final_sets[display_id] = set(id_set)
        
        # 2. ID 전환 관계로 연결된 집합 병합
        processed = set()
        merged_sets = {}
        
        for display_id, id_set in sorted(final_sets.items()):
            if display_id in processed:
                continue
                
            # 현재 집합에서 시작해 연결된 모든 ID 찾기
            merged_id_set = set(id_set)
            queue = list(id_set)
            related_display_ids = {display_id}
            
            while queue:
                current_id = queue.pop(0)
                
                # 현재 ID와 전환 관계가 있는 모든 ID 확인
                if current_id in self.id_transitions:
                    for related_id in self.id_transitions[current_id]:
                        # 해당 ID가 어떤 집합에 속하는지 확인
                        for other_display_id, other_id_set in final_sets.items():
                            if related_id in other_id_set and other_display_id not in processed:
                                # 새로운 ID 집합을 현재 병합 집합에 추가
                                new_ids = other_id_set - merged_id_set
                                merged_id_set.update(new_ids)
                                queue.extend(list(new_ids))
                                related_display_ids.add(other_display_id)
            
            # 병합된 ID 집합에서 최소 ID를 새로운 대표 ID로 설정
            min_display_id = min(related_display_ids)
            merged_sets[min_display_id] = merged_id_set
            processed.update(related_display_ids)
        
        return merged_sets
    
    def _calculate_scene_complexity(self, boxes, frame_shape):
        """장면 복잡도 계산"""
        if not boxes:
            return 0.0
        
        # 객체 밀도
        object_density = min(1.0, len(boxes) / 20.0)
        
        # 객체 간 평균 거리
        all_centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]
        avg_distance = 0.0
        
        if len(all_centers) > 1:
            distances = []
            for i in range(len(all_centers)):
                for j in range(i+1, len(all_centers)):
                    dist = ((all_centers[i][0] - all_centers[j][0]) ** 2 + 
                           (all_centers[i][1] - all_centers[j][1]) ** 2) ** 0.5
                    distances.append(dist)
            
            avg_distance = sum(distances) / len(distances) if distances else float('inf')
            normalized_distance = max(0.0, min(1.0, 1.0 - (avg_distance / 200.0)))
        else:
            normalized_distance = 0.0
        
        # 복잡도 = 밀도 * 0.7 + 정규화된 거리 * 0.3
        complexity = object_density * 0.7 + normalized_distance * 0.3
        
        return complexity
    
    def _update_adaptive_threshold(self, base_threshold, complexity, avg_speed, confidence_level):
        """적응형 임계값 계산"""
        # 복잡도가 높을수록 임계값 감소
        complexity_factor = 1.0 - (complexity * 0.5)
        
        # 속도가 빠를수록 임계값 증가
        speed_factor = 1.0 + min(1.0, avg_speed / 10.0) * 0.5
        
        # 신뢰도 수준에 따른 조정
        confidence_factor = 0.8 + (confidence_level * 0.4)
        
        # 최종 임계값 계산 및 제한
        threshold = base_threshold * complexity_factor * speed_factor * confidence_factor
        return max(5.0, min(20.0, threshold))
    
    def _calculate_iou(self, box1, box2):
        """두 바운딩 박스의 IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_histograms(self, hist1, hist2):
        """두 히스토그램 간의 유사도 계산"""
        if hist1 is None or hist2 is None:
            return 0.0
            
        try:
            # 바타차랴 거리 계산
            bc_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            
            # 유사도로 변환 (1 - 거리)
            similarity = 1.0 - bc_distance
            
            return similarity
        except:
            return 0.0
    
    def _is_consistent_movement(self, obj1, obj2):
        """두 객체의 움직임 일관성 확인"""
        # 경로 길이 확인
        hist1 = obj1.centers_history
        hist2 = obj2.centers_history
        
        if len(hist1) < 3 or len(hist2) < 3:
            return True  # 충분한 데이터 없으면 일관성 있다고 가정
        
        # 이동 벡터 계산
        vec1 = (hist1[-1][0] - hist1[-2][0], hist1[-1][1] - hist1[-2][1])
        vec2 = (hist2[-1][0] - hist2[-2][0], hist2[-1][1] - hist2[-2][1])
        
        # 벡터 크기 검사
        vec1_magnitude = (vec1[0]**2 + vec1[1]**2)**0.5
        vec2_magnitude = (vec2[0]**2 + vec2[1]**2)**0.5
        
        if vec1_magnitude < 3 or vec2_magnitude < 3:
            return True  # 정지 상태면 일관성 있다고 가정
        
        # 코사인 유사도 계산
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        cosine_similarity = dot_product / (vec1_magnitude * vec2_magnitude)
        
        return cosine_similarity > 0.7  # 45도 이내면 일관적으로 간주
    
    def _print_debug_info(self):
        """디버그 정보 출력"""
        print("\n==== ID Manager 상태 정보 ====")
        print(f"프레임: {self.frame_count}, 추적 객체: {len(self.tracked_objects)}")
        print(f"장면 복잡도: {self.scene_complexity:.2f}, 평균 속도: {self.avg_object_speed:.2f}")
        print(f"현재 거리 임계값: {self.current_distance_threshold:.2f}")
        
        # 활성 ID 집합 정보
        print("활성 ID 집합 (크기>1):")
        for i, id_set in enumerate(self.chicken_id_sets):
            if len(id_set) > 1:
                display_id = self.set_to_display_id.get(i, "없음")
                print(f"  집합 {i}: 대표ID={display_id}, IDs={sorted(id_set)}")
        
        # 메모리 사용량
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"메모리 사용량: {mem_info.rss / 1024 / 1024:.1f} MB")
        print("=" * 30)


def main():
    """메인 함수: YOLO 객체 탐지 및 추적 실행"""
    # 파일 경로 설정
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
    model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best.pt"
    result_file = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\tracking_results.txt"
    
    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return
      
    # YOLO 모델 로드
    try:
        model = YOLO(model_path)
        print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    
    # 화면 창 생성
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection", int(width * 0.3), int(height * 0.3))
    
    # ID 관리자 초기화
    id_manager = IDManager({
        'base_distance_threshold': 10,
        'iou_threshold': 0.2,
        'size_difference_threshold': 1.5,
        'histogram_similarity_threshold': 0.6,
        'max_frames': 100,
        'max_trail_length': 30
    })
    id_manager.verbose = True  # 디버그 정보 출력 활성화
    
    # 탐지 설정
    conf_threshold = 0.5
    frame_count = 0
    chicken_class_id = 0  # 닭 클래스 ID
    
    # FPS 계산용 변수
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0
    
    # 시간 제한 설정
    start_time = 0  # 시작 시간 (초)
    end_time = 60   # 종료 시간 (초)
    
    # 메인 루프
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("비디오의 끝에 도달했습니다.")
            break
        
        # 프레임 카운터 증가
        frame_count += 1
        
        # 비디오 현재 시간 계산 (초 단위)
        video_time = frame_count / fps
        
        # 지정된 시간 범위 밖이면 건너뛰기
        if video_time < start_time:
            continue
        if end_time > 0 and video_time > end_time:
            print(f"지정된 종료 시간({end_time}초)에 도달했습니다.")
            break
        
        # FPS 계산
        current_time = time.time()
        fps_frame_count += 1
        if current_time - fps_start_time >= 1.0:
            fps_display = fps_frame_count / (current_time - fps_start_time)
            fps_frame_count = 0
            fps_start_time = current_time
        
        # 관심 영역만 처리 (필요한 부분만 잘라서 사용)
        frame = frame[1000:1500, 200:1000]
        
        # 객체 탐지 수행
        results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)[0]
        
        # 탐지 결과 추출
        boxes = results.boxes.xyxy.cpu().tolist() if hasattr(results.boxes, 'xyxy') else []
        ids = results.boxes.id.int().cpu().tolist() if hasattr(results.boxes, 'id') and results.boxes.id is not None else [-1] * len(boxes)
        classes = results.boxes.cls.int().cpu().tolist() if hasattr(results.boxes, 'cls') else [chicken_class_id] * len(boxes)
        
        # 닭 클래스만 필터링
        filtered_indices = [i for i, cls in enumerate(classes) if cls == chicken_class_id]
        boxes = [boxes[i] for i in filtered_indices]
        ids = [ids[i] for i in filtered_indices]
        
        # 박스와 ID 결합
        boxes_with_ids = [(box[0], box[1], box[2], box[3], id) for box, id in zip(boxes, ids)]
        
        # ID 관리자 업데이트
        boxes_with_display_ids, changes = id_manager.update(boxes_with_ids, frame, current_time)
        
        # 결과 시각화
        display_frame = frame.copy()
        
        # 객체별 시각화
        for box_data in boxes_with_display_ids:
            x1, y1, x2, y2, obj_id, display_id = box_data
            
            # 객체 상태 파악 (새로 할당된 ID인지, ID가 변경된 적이 있는지)
            is_reassigned = (obj_id != display_id and obj_id != -1)
            id_set = id_manager.get_id_set(obj_id) if obj_id != -1 else set()
            id_changed = len(id_set) > 1
            
            # 색상 결정
            if is_reassigned:
                box_color = (0, 165, 255)  # 주황색 (BGR)
            elif id_changed:
                box_color = (0, 255, 0)    # 녹색 (BGR)
            else:
                box_color = (0, 200, 0)    # 녹색 (BGR)
            
            # 바운딩 박스 그리기
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            
            # ID 표시 텍스트 생성
            if obj_id != -1:
                if len(id_set) > 1:
                    if len(id_set) <= 5:
                        id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                    else:
                        sorted_ids = sorted(id_set)
                        id_set_str = f"{{{sorted_ids[0]}...{sorted_ids[-1]}}}"
                    
                    if is_reassigned:
                        label = f"{obj_id}→{display_id} {id_set_str}"
                    else:
                        label = f"{display_id} {id_set_str}"
                else:
                    label = f"{display_id}"
            else:
                label = "?"
            
            # 배경 박스
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # ID 상태에 따라 배경색 조정
            if is_reassigned:
                bg_color = (50, 50, 0)  # 어두운 주황색 배경
            elif id_changed:
                bg_color = (0, 50, 0)  # 어두운 녹색 배경
            else:
                bg_color = (0, 0, 0)  # 검은색 배경
            
            cv2.rectangle(display_frame, 
                         (int(x1), int(y1) - text_height - 5), 
                         (int(x1) + text_width + 5, int(y1)), 
                         bg_color, -1)
            
            # 텍스트 표시
            cv2.putText(display_frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 객체 경로 표시
            if obj_id != -1:
                trail = id_manager.get_object_trail(obj_id)
                if len(trail) > 1:
                    # 고유한 색상 생성
                    color_base = (display_id * 50) % 256
                    trail_color = (color_base, 255, 255-color_base)
                    
                    # 경로 그리기
                    for i in range(1, len(trail)):
                        pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                        pt2 = (int(trail[i][0]), int(trail[i][1]))
                        thickness = 1 + int((i / len(trail)) * 2)
                        cv2.line(display_frame, pt1, pt2, trail_color, thickness)
        
        # 정보 표시
        cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "ID 형식: 현재ID→대표ID {ID 집합}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
        cv2.putText(display_frame, f"탐지된 객체 수: {len(boxes_with_display_ids)}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
        
        # 화면에 표시
        cv2.imshow("Object Detection", display_frame)
        
        # 키 입력 처리
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print("사용자가 종료했습니다.")
            break
        elif key == ord(' '):
            print("일시정지됨. 계속하려면 아무 키나 누르세요.")
            cv2.waitKey(0)
    
    # 결과 저장
    print(f"추적 결과를 {result_file}에 저장 중...")
    id_manager.save_id_sets_to_file(result_file)
    print(f"추적 결과가 성공적으로 저장되었습니다.")
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()