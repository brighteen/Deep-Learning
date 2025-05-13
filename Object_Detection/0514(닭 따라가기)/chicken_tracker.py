import pandas as pd
import time
from datetime import datetime
import cv2
import numpy as np
import os

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
    
    def export_to_csv(self, filename=None, fps=30):
        """
        닭 ID 변화 정보만 CSV 파일로 내보냅니다.
        
        Args:
            filename: 저장할 파일명 (None이면 자동 생성)
            fps: 영상의 FPS (초당 프레임 수)
            
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
        
        try:
            # ID 변화 데이터만 추출
            id_changes = []
            # 프레임별로 어떤 ID가 등장했는지 추적
            frame_ids = {}
            # ID 변화 추적을 위한 마지막 상태 기록
            last_status = {}
            
            for item in self.history:
                frame = item['frame']
                chicken_id = item['chicken_id']
                status = item['status']
                set_id = item['set_id']
                
                # ID가 유효하지 않으면 건너뜀
                if chicken_id == -1:
                    continue
                    
                # 상태 변화가 있는 경우만 기록
                current_state = (status, set_id)
                if chicken_id not in last_status or last_status[chicken_id] != current_state:
                    # 프레임별로 ID 목록 정리
                    if frame not in frame_ids:
                        frame_ids[frame] = {}
                    
                    # 변화된 ID 정보 기록
                    frame_ids[frame][chicken_id] = {
                        'status': status,
                        'set_id': set_id
                    }
                    
                    # 현재 상태를 마지막 상태로 업데이트
                    last_status[chicken_id] = current_state
            
            if not frame_ids:
                print("저장할 ID 변화 데이터가 없습니다.")
                # 기본 데이터라도 저장
                df = pd.DataFrame(columns=['frame', 'time', 'changed_ids', 'sets', 'status'])
                df.to_csv(filename, index=False)
                print(f"빈 CSV 파일이 생성되었습니다: {filename}")
                return filename
                
            # 각 프레임에 ID 변화가 있는 경우만 기록
            for frame, ids in sorted(frame_ids.items()):
                time_seconds = frame / fps
                
                # 변환할 데이터 준비
                changed_ids = list(ids.keys())
                sets = [ids[id_key]['set_id'] for id_key in ids.keys()]
                status_dict = {str(id_key): data['status'] for id_key, data in ids.items()}
                
                # ID 변화 정보만 포함
                id_changes.append({
                    'frame': frame,
                    'time': time_seconds,
                    'changed_ids': str(changed_ids),  # 리스트를 문자열로 저장
                    'sets': str(sets),                # 리스트를 문자열로 저장
                    'status': str(status_dict)        # 딕셔너리를 문자열로 저장
                })
            
            # DataFrame 생성 및 저장
            df = pd.DataFrame(id_changes)
            df.to_csv(filename, index=False)
            print(f"CSV 파일이 성공적으로 저장되었습니다: {filename}")
            
            return filename
            
        except Exception as e:
            print(f"CSV 파일 저장 중 오류 발생: {e}")
            # 오류 정보 저장
            error_log = os.path.splitext(filename)[0] + "_error.txt"
            with open(error_log, 'w') as f:
                f.write(f"CSV 저장 중 오류: {str(e)}\n")
                f.write(f"시간: {datetime.now()}\n")
                f.write(f"데이터 항목 수: {len(self.history)}\n")
            
            return None

    def visualize_tracking(self, frame, results):
        """
        추적 결과를 시각화합니다.
        
        Args:
            frame: 시각화할 프레임
            results: YOLOv8 추적 결과
            
        Returns:
            numpy.ndarray: 시각화된 프레임
        """
        # 기본 시각화 (YOLO가 제공하는 plot 메소드 사용)
        plotted_frame = results.plot(labels=True, font_size=0.5, line_width=1)
        
        # ID 집합 번호 추가
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().tolist()
            ids = results.boxes.id.int().cpu().tolist()
            
            for i, box in enumerate(boxes):
                if i < len(ids):
                    x1, y1, x2, y2 = map(int, box)
                    chicken_id = ids[i]
                    set_id = self._get_set_id(chicken_id)
                    cv2.putText(plotted_frame, f"S{set_id}", (x1, y1-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return plotted_frame

    def print_stats(self):
        """
        추적 통계를 출력합니다.
        """
        stats = self.get_tracking_stats()
        print("\n=== 닭 추적 통계 ===")
        print(f"총 ID 집합 수: {stats['id_sets']}개")
        print(f"총 고유 ID 수: {stats['total_unique_ids']}개")
        print(f"재등장 이벤트 수: {stats['reappearance_events']}회")
        print("\n=== ID 집합별 세부 정보 ===")
        for set_name, set_info in stats['set_details'].items():
            print(f"{set_name}: {set_info['ids']} (총 {set_info['count']}개)")
        
        # 각 닭의 ID 변화 출력
        self.print_chicken_id_changes()
    
    def print_chicken_id_changes(self):
        """
        각 닭의 ID 변화를 출력합니다.
        예: 닭 1: [1, 11, 13]
        """
        if not self.id_sets:
            print("추적된 닭 ID가 없습니다.")
            return
        
        print("\n=== 각 닭의 ID 변화 ===")
        for i, id_set in enumerate(self.id_sets):
            # 집합 내의 ID를 정렬하여 출력
            sorted_ids = sorted(list(id_set))
            print(f"닭 {i+1}: {sorted_ids}")
        
        return
