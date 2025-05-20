"""
ID 관리 모듈

객체 추적 시스템에서 ID를 효율적으로 관리하기 위한 클래스 구현

이 모듈은 객체가 화면에서 사라졌다가 다시 나타나도 동일한 ID를 유지하도록
거리와 시간 기반의 매칭 알고리즘을 제공합니다.

원본 코드를 경량화하고 개선하였습니다.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional


@dataclass
class TrackedObject:
    """
    추적 객체 정보를 저장하는 클래스
    
    객체의 ID, 위치, 등장 시간 등 모든 관련 정보를 하나의 클래스에 통합하여 관리
    """
    id: int                     # 객체 ID
    position: Tuple[float, float]  # 위치 좌표 (x, y)
    last_frame: int             # 마지막으로 등장한 프레임 번호
    last_time: float            # 마지막으로 등장한 시간
    set_index: Optional[int] = None  # 소속 집합 인덱스


def calculate_center(box):
    """
    바운딩 박스의 중심 좌표를 계산하는 함수
    
    Args:
        box: 바운딩 박스 좌표 [x1, y1, x2, y2]
        
    Returns:
        (center_x, center_y): 박스 중심의 x, y 좌표 튜플
    """
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def calculate_distance(center1, center2):
    """
    두 중심점 간의 유클리디안 거리를 계산하는 함수
    
    Args:
        center1: 첫 번째 중심점 좌표 (x, y)
        center2: 두 번째 중심점 좌표 (x, y)
        
    Returns:
        두 점 사이의 유클리디안 거리
    """
    return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5


class IDManager:
    """
    객체 ID를 일관되게 관리하는 클래스 (경량화 버전)
    
    같은 객체의 다양한 ID를 추적하고 관리하기 위한 클래스입니다.
    객체가 프레임에서 사라졌다가 다시 나타나도 일관된 ID를 유지하도록 합니다.
    """
    
    def __init__(self, distance_threshold=25, time_threshold=1.5, max_frames=100):
        """
        ID 관리자 객체를 초기화합니다.
        
        Args:
            distance_threshold: 같은 객체로 판단할 최대 거리 (픽셀 단위), 기본값 25
            time_threshold: 같은 객체로 판단할 최대 시간 차이 (초 단위), 기본값 1.5
            max_frames: 오래된 데이터 삭제 기준 프레임 수, 기본값 100
        """
        # 핵심 데이터 구조 (경량화)
        self.tracked_objects = {}  # id -> TrackedObject 매핑
        self.id_sets = {}          # 집합 인덱스 -> ID 집합 매핑
        self.set_display_ids = {}  # 집합 인덱스 -> 대표 ID 매핑
        
        # 설정 값
        self.distance_threshold = distance_threshold
        self.time_threshold = time_threshold
        self.max_frames = max_frames
        
        # 상태 추적
        self.frame_count = 0
        self.next_set_id = 0  # 다음 집합 ID (안정적인 인덱싱을 위해)
        
        # 메모리 모니터링
        self.mem_clean_trigger = 1000  # 메모리 정리 트리거 카운트
        self.last_clean_frame = 0      # 마지막 메모리 정리 프레임
        
    def update_id_mapping(self, boxes, current_time):
        """
        탐지된 객체들의 ID 매핑을 업데이트합니다.
        
        Args:
            boxes: 탐지된 경계 상자들 목록 [(x1, y1, x2, y2, id), ...]
            current_time: 현재 시간 (초 단위)
            
        Returns:
            매핑된 ID 목록 (원본 ID 순서와 동일)
        """
        # 프레임 카운터 증가
        self.frame_count += 1
        
        # 현재 프레임에서 탐지된 객체 정보 추출
        current_boxes = [(box[0], box[1], box[2], box[3]) for box in boxes]
        current_ids = [int(box[4]) for box in boxes]
        current_id_set = set(current_ids)
        
        # 모든 바운딩 박스의 중심점 계산
        centers = [calculate_center(box) for box in current_boxes]
        
        # 매칭 추적을 위한 변수
        already_matched_old_ids = set()
        already_matched_new_ids = set()
        
        # 최근에 사라진 ID 식별 
        # (현재 프레임에 없지만 time_threshold 이내에 사라진 ID들)
        recently_disappeared_ids = {}
        
        for old_id, obj in self.tracked_objects.items():
            if old_id not in current_id_set:
                time_diff = current_time - obj.last_time
                
                if time_diff < self.time_threshold:
                    recently_disappeared_ids[old_id] = (obj.position, time_diff)
        
        # 새로운 ID와 최근에 사라진 ID 사이의 매칭 계산
        potential_matches = []
        
        for i, chicken_id in enumerate(current_ids):
            center = centers[i]
            
            # 이미 알고 있는 ID인 경우 위치 정보 업데이트
            if chicken_id in self.tracked_objects:
                obj = self.tracked_objects[chicken_id]
                obj.position = center
                obj.last_frame = self.frame_count
                obj.last_time = current_time
                already_matched_new_ids.add(chicken_id)
                continue
                
            # 새로운 ID가 나타난 경우 최근에 사라진 ID와 매칭 시도
            for old_id, (old_pos, time_diff) in recently_disappeared_ids.items():
                # 이미 매칭된 ID는 건너뜀
                if old_id in already_matched_old_ids:
                    continue
                    
                # 거리 계산 (제곱 형태로 저장하여 최적화)
                distance_squared = (center[0] - old_pos[0])**2 + (center[1] - old_pos[1])**2
                
                # 거리가 임계값 이내인 경우
                if distance_squared < (self.distance_threshold**2):
                    # 매칭 점수 계산 (거리와 시간 가중치 조합)
                    match_score = distance_squared + (time_diff * 10)
                    potential_matches.append((match_score, i, chicken_id, old_id))
        
        # 매칭 점수가 낮은 순서로 정렬 (가장 좋은 매칭부터 처리)
        potential_matches.sort()
        
        for score, i, chicken_id, old_id in potential_matches:
            # 이미 매칭된 ID는 건너뜀
            if old_id in already_matched_old_ids or chicken_id in already_matched_new_ids:
                continue
                  
            # 매칭 성공 처리
            already_matched_old_ids.add(old_id)
            already_matched_new_ids.add(chicken_id)
            center = centers[i]
            
            # ID 집합 관리
            old_obj = self.tracked_objects[old_id]
            set_index = old_obj.set_index
            
            if set_index is not None and set_index in self.id_sets:
                # 기존 집합에 새 ID 추가
                self.id_sets[set_index].add(chicken_id)
                
                # 새 객체 생성 및 추가
                new_obj = TrackedObject(
                    id=chicken_id,
                    position=center,
                    last_frame=self.frame_count,
                    last_time=current_time,
                    set_index=set_index
                )
                self.tracked_objects[chicken_id] = new_obj
            else:
                # 새 집합 생성
                set_index = self.next_set_id
                self.next_set_id += 1
                
                self.id_sets[set_index] = {old_id, chicken_id}
                self.set_display_ids[set_index] = min(chicken_id, old_id)
                
                # 기존 객체 업데이트
                old_obj.set_index = set_index
                
                # 새 객체 생성 및 추가
                new_obj = TrackedObject(
                    id=chicken_id,
                    position=center,
                    last_frame=self.frame_count,
                    last_time=current_time,
                    set_index=set_index
                )
                self.tracked_objects[chicken_id] = new_obj
        
        # 매칭되지 않은 새로운 ID들을 처리
        for i, chicken_id in enumerate(current_ids):
            if chicken_id not in already_matched_new_ids:
                # 새 집합 생성
                set_index = self.next_set_id
                self.next_set_id += 1
                
                self.id_sets[set_index] = {chicken_id}
                self.set_display_ids[set_index] = chicken_id
                
                # 새 객체 생성
                new_obj = TrackedObject(
                    id=chicken_id,
                    position=centers[i],
                    last_frame=self.frame_count,
                    last_time=current_time,
                    set_index=set_index
                )
                self.tracked_objects[chicken_id] = new_obj
        
        # 메모리 관리: 오래된 데이터 정리
        if self.frame_count - self.last_clean_frame >= self.mem_clean_trigger:
            self.clean_up_stale_data()
            self.last_clean_frame = self.frame_count
            
        # 매핑된 ID 목록 생성 (원본 ID 순서와 동일)
        mapped_ids = [self.get_display_id(id) for id in current_ids]
        return mapped_ids
    
    def get_display_id(self, chicken_id):
        """
        특정 ID에 대한 대표 ID를 반환합니다.
        
        Args:
            chicken_id: 대표 ID를 찾을 객체 ID
            
        Returns:
            대표 ID (해당 집합의 대표 ID 또는 입력 ID 자체)
        """
        # 객체가 추적 중이고 유효한 집합에 속해 있는지 확인
        if chicken_id in self.tracked_objects:
            obj = self.tracked_objects[chicken_id]
            if obj.set_index is not None and obj.set_index in self.set_display_ids:
                return self.set_display_ids[obj.set_index]
        
        # 집합이 없거나 유효하지 않은 경우 입력 ID 그대로 반환
        return chicken_id
    
    def get_id_set(self, chicken_id):
        """
        특정 ID가 속한 ID 집합을 반환합니다.
        
        Args:
            chicken_id: 집합을 찾을 객체 ID
            
        Returns:
            ID 집합 (ID가 유효한 집합에 속해 있으면 해당 집합, 아니면 ID 자체를 포함하는 집합)
        """
        # 객체가 추적 중이고 유효한 집합에 속해 있는지 확인
        if chicken_id in self.tracked_objects:
            obj = self.tracked_objects[chicken_id]
            if obj.set_index is not None and obj.set_index in self.id_sets:
                return self.id_sets[obj.set_index]
        
        # 집합이 없거나 유효하지 않은 경우 단일 ID 집합 반환
        return {chicken_id}
    
    def clean_up_stale_data(self):
        """
        오래된 추적 데이터를 정리합니다.
        
        메모리 관리를 위해 오래된 ID 데이터와 비어있는 집합을 정리합니다.
        """
        # 프레임 임계값 설정
        frame_threshold = self.frame_count - self.max_frames
        
        # 오래된 ID 식별
        stale_ids = [obj.id for obj in self.tracked_objects.values() 
                    if obj.last_frame <= frame_threshold]
        
        # 삭제할 ID가 없으면 반환
        if not stale_ids:
            return
        
        # 영향받는 집합 추적
        affected_sets = set()
        
        # 오래된 객체 삭제
        for chicken_id in stale_ids:
            if chicken_id in self.tracked_objects:
                obj = self.tracked_objects[chicken_id]
                if obj.set_index is not None:
                    # 집합에서 ID 제거
                    if obj.set_index in self.id_sets:
                        self.id_sets[obj.set_index].discard(chicken_id)
                        affected_sets.add(obj.set_index)
                
                # 객체 삭제
                del self.tracked_objects[chicken_id]
        
        # 빈 집합 식별 및 정리
        empty_set_indices = [idx for idx in affected_sets 
                            if idx in self.id_sets and not self.id_sets[idx]]
        
        # 빈 집합 정리
        for set_index in empty_set_indices:
            if set_index in self.id_sets:
                del self.id_sets[set_index]
            if set_index in self.set_display_ids:
                del self.set_display_ids[set_index]
    
    def save_id_sets_to_file(self, filename):
        """
        ID 집합을 파일로 저장합니다.
        
        Args:
            filename: 저장할 파일 경로
            
        Returns:
            저장된 최종 ID 집합 딕셔너리 {대표 ID: {ID 집합}, ...}
        """
        # 최종 ID 집합을 저장할 딕셔너리
        final_id_sets = {}
        
        # 모든 ID 집합을 처리
        for set_index, id_set in self.id_sets.items():
            # 빈 집합은 건너뜀
            if not id_set:
                continue
                
            # 이 집합의 대표 ID 가져오기
            display_id = self.set_display_ids.get(set_index, min(id_set) if id_set else None)
            
            if display_id is not None:
                # 동일한 대표 ID를 가진 집합이 이미 있으면 병합
                if display_id in final_id_sets:
                    final_id_sets[display_id].update(id_set)
                else:
                    # 새 대표 ID인 경우 새 집합으로 추가
                    final_id_sets[display_id] = set(id_set)
        
        # 파일에 저장 시 예외 처리 추가
        try:
            with open(filename, 'w') as f:
                # 대표 ID를 기준으로 정렬하여 가독성 향상
                for display_id in sorted(final_id_sets.keys()):
                    id_set = final_id_sets[display_id]
                    # 비어있지 않은 집합만 저장
                    if id_set:
                        # ID 집합을 정렬된 문자열로 변환
                        id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                        # "대표ID: {ID1, ID2, ID3, ...}" 형식으로 저장
                        f.write(f"{display_id}: {id_set_str}\n")
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
        
        # 변환된 최종 ID 집합 반환
        return final_id_sets
    
    def get_status_info(self):
        """
        현재 ID 관리 상태 정보를 반환합니다.
        
        Returns:
            상태 정보 딕셔너리
        """
        return {
            "tracked_objects_count": len(self.tracked_objects),
            "id_sets_count": len(self.id_sets),
            "frame_count": self.frame_count,
            "memory_clean_trigger": self.mem_clean_trigger,
            "last_clean_frame": self.last_clean_frame
        }


def main():
    """
    ID 관리 로직을 테스트하는 메인 함수
    
    이 함수는 IDManager 클래스의 기능을 시연하기 위한 몇 가지 시나리오를 실행합니다:
    1. 객체 등장: 초기에 3개의 객체가 서로 다른 위치에 등장
    2. 객체 사라짐/새 객체 등장: 일부 객체가 사라지고 비슷한 위치에 새 객체 등장
    3. 재등장 감지: 이전에 사라진 객체가 다시 나타나고, 새로운 객체도 등장
    
    각 단계에서 ID 집합이 어떻게 형성되고 관리되는지 보여줍니다.
    """
    # ID 관리자 객체 생성 (파라미터 명시)
    id_manager = IDManager(
        distance_threshold=25,  # 거리 임계값
        time_threshold=1.5,     # 시간 임계값
        max_frames=100          # 프레임 임계값
    )
    
    # 테스트 시나리오 1: 초기 객체 등장
    print("\n===== 시나리오 1: 초기 객체 등장 =====")
    test_boxes = [
        [100, 100, 200, 200, 1],  # 객체 ID 1의 위치
        [300, 300, 400, 400, 2],  # 객체 ID 2의 위치
        [500, 500, 600, 600, 3]   # 객체 ID 3의 위치
    ]
    
    # 첫 번째 프레임 데이터 업데이트
    mapped_ids = id_manager.update_id_mapping(test_boxes, time.time())
    print(f"프레임 1 후 매핑된 ID: {mapped_ids}")
    print(f"ID 집합: {list(id_manager.id_sets.values())}")
    
    # 객체 상태 출력
    print(f"추적 객체 수: {len(id_manager.tracked_objects)}")
    
    # 1초 대기 (시간 경과 시뮬레이션)
    time.sleep(1)
    
    # 테스트 시나리오 2: 객체 사라짐 및 새 객체 등장
    print("\n===== 시나리오 2: 객체 사라짐 및 새 객체 등장 =====")
    test_boxes = [
        [305, 305, 405, 405, 4],  # ID 2와 비슷한 위치에 새로운 ID 4가 등장 (ID 2와 매칭 예상)
        [510, 510, 610, 610, 3]   # ID 3은 약간 이동 (계속 추적)
        # ID 1은 사라짐
    ]
    
    # 두 번째 프레임 데이터 업데이트
    mapped_ids = id_manager.update_id_mapping(test_boxes, time.time())
    print(f"프레임 2 후 매핑된 ID: {mapped_ids}")
    print(f"ID 집합: {list(id_manager.id_sets.values())}")
    
    # 각 ID에 대한 실제 매핑 확인
    for id in [3, 4]:
        print(f"ID {id}의 대표 ID: {id_manager.get_display_id(id)}, 소속 집합: {id_manager.get_id_set(id)}")
    
    # 1초 대기 (시간 경과 시뮬레이션)
    time.sleep(1)
    
    # 테스트 시나리오 3: 이전 객체 재등장 및 추가 객체 등장
    print("\n===== 시나리오 3: 이전 객체 재등장 및 추가 객체 등장 =====")
    test_boxes = [
        [110, 110, 210, 210, 1],  # ID 1이 거의 같은 위치에 다시 등장 (ID 1로 인식 예상)
        [307, 307, 407, 407, 4],  # ID 4는 약간 이동 (계속 추적)
        [515, 515, 615, 615, 5]   # ID 3과 비슷한 위치에 새로운 ID 5 등장 (ID 3과 매칭 예상)
    ]
    
    # 세 번째 프레임 데이터 업데이트
    mapped_ids = id_manager.update_id_mapping(test_boxes, time.time())
    print(f"프레임 3 후 매핑된 ID: {mapped_ids}")
    print(f"ID 집합: {list(id_manager.id_sets.values())}")
    
    # ID 집합을 파일로 저장하고 최종 결과 출력
    print("\n===== 최종 ID 매핑 결과 =====")
    final_sets = id_manager.save_id_sets_to_file("id_sets_test.txt")
    print("최종 ID 집합 (저장됨):", final_sets)
    print("파일에 저장됨: id_sets_test.txt")
    
    # 각 ID의 대표 ID 및 소속 집합 확인
    print("\n===== 각 ID의 대표 ID 및 소속 집합 =====")
    for id in [1, 2, 3, 4, 5]:
        display_id = id_manager.get_display_id(id)
        id_set = id_manager.get_id_set(id)
        print(f"ID {id}의 대표 ID: {display_id}, 소속 집합: {id_set}")
    
    # 현재 상태 정보 출력
    print("\n===== ID 관리자 상태 정보 =====")
    status = id_manager.get_status_info()
    for key, value in status.items():
        print(f"{key}: {value}")


# 스크립트 직접 실행 시 메인 함수 호출
if __name__ == "__main__":
    main()
