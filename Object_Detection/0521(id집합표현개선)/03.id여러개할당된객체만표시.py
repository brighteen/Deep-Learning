import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

class IDManager:
    """객체 ID를 관리하는 간소화된 클래스"""
    
    def __init__(self):
        # ID 집합 관리 데이터
        self.id_sets = []  # 같은 개체로 판단된 ID 집합들
        self.id_to_set_index = {}  # ID가 어느 집합에 속하는지 매핑
        self.set_to_display_id = {} # 각 집합의 대표 ID
        self.id_to_box = {}  # 각 ID의 바운딩 박스 정보
        self.frame_count = 0  # 현재 프레임 번호
        self.all_detected_ids = set()  # 이전에 탐지된 모든 ID
        
        # 설정
        self.distance_threshold = 11  # 같은 객체로 판단할 최대 거리
    
    def update(self, detected_objects):
        """
        탐지된 객체 정보 업데이트 및 ID 집합 관리
        
        Args:
            detected_objects: [(x1, y1, x2, y2, id), ...] 형태의 객체 목록
        """
        self.frame_count += 1
        
        # 탐지된 객체 정보 저장
        for obj in detected_objects:
            x1, y1, x2, y2, obj_id = obj
            box = (x1, y1, x2, y2)
            
            # ID가 유효한 경우에만 처리
            if obj_id != -1:
                # 모든 탐지된 ID 기록
                self.all_detected_ids.add(obj_id)
                
                # 바운딩 박스 정보 저장
                self.id_to_box[obj_id] = box
                
                # 해당 ID가 어떤 집합에 속하는지 확인
                if obj_id in self.id_to_set_index:
                    set_index = self.id_to_set_index[obj_id]
                else:
                    # 새 집합 생성
                    set_index = len(self.id_sets)
                    self.id_sets.append({obj_id})
                    self.id_to_set_index[obj_id] = set_index
                    self.set_to_display_id[set_index] = obj_id
        
        # ID 집합 병합 처리
        self._merge_id_sets(detected_objects)
        
        # 한 번도 집합에 포함되지 않은 ID 처리
        self._ensure_all_ids_in_sets()
    
    def _ensure_all_ids_in_sets(self):
        """모든 탐지된 ID가 어떤 집합에 속하도록 보장"""
        for obj_id in self.all_detected_ids:
            if obj_id not in self.id_to_set_index:
                # 집합에 포함되지 않은 ID는 새 집합 생성
                set_index = len(self.id_sets)
                self.id_sets.append({obj_id})
                self.id_to_set_index[obj_id] = set_index
                self.set_to_display_id[set_index] = obj_id
    
    def _merge_id_sets(self, objects):
        """서로 가까운 객체들의 ID 집합을 병합"""
        # 각 객체 쌍에 대해 거리 계산
        for i, obj1 in enumerate(objects):
            x1_1, y1_1, x2_1, y2_1, id1 = obj1
            if id1 == -1:
                continue
                
            center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
            
            for j in range(i + 1, len(objects)):
                x1_2, y1_2, x2_2, y2_2, id2 = objects[j]
                if id2 == -1:
                    continue
                    
                # 이미 같은 집합인 경우 건너뜀
                if (id1 in self.id_to_set_index and id2 in self.id_to_set_index and 
                    self.id_to_set_index[id1] == self.id_to_set_index[id2]):
                    continue
                
                # 두 객체 간 거리 계산
                center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                
                # 거리가 임계값보다 가까우면 두 집합 병합
                if distance < self.distance_threshold:
                    set_index1 = self.id_to_set_index.get(id1)
                    set_index2 = self.id_to_set_index.get(id2)
                    
                    if set_index1 is not None and set_index2 is not None:
                        if set_index1 != set_index2:
                            # 작은 인덱스의 집합으로 병합
                            from_index = max(set_index1, set_index2)
                            to_index = min(set_index1, set_index2)
                            
                            # 집합 병합
                            self.id_sets[to_index].update(self.id_sets[from_index])
                            
                            # ID와 집합 인덱스 매핑 업데이트
                            for moved_id in self.id_sets[from_index]:
                                self.id_to_set_index[moved_id] = to_index
                            
                            # 대표 ID 업데이트 (작은 ID를 대표로 사용)
                            self.set_to_display_id[to_index] = min(
                                self.set_to_display_id.get(to_index, float('inf')),
                                self.set_to_display_id.get(from_index, float('inf'))
                            )
                            
                            # 병합된 집합 비우기
                            self.id_sets[from_index] = set()
                            self.set_to_display_id.pop(from_index, None)
                    
                    elif set_index1 is not None:
                        # id2를 id1의 집합에 추가
                        self.id_sets[set_index1].add(id2)
                        self.id_to_set_index[id2] = set_index1
                    
                    elif set_index2 is not None:
                        # id1을 id2의 집합에 추가
                        self.id_sets[set_index2].add(id1)
                        self.id_to_set_index[id1] = set_index2
    
    def get_display_id(self, obj_id):
        """객체의 대표 ID 반환"""
        if obj_id in self.id_to_set_index:
            set_index = self.id_to_set_index[obj_id]
            return self.set_to_display_id.get(set_index, obj_id)
        return obj_id
    
    def get_id_set(self, obj_id):
        """객체가 속한 ID 집합 반환"""
        if obj_id in self.id_to_set_index:
            set_index = self.id_to_set_index[obj_id]
            return self.id_sets[set_index]
        return {obj_id}
    
    def save_id_sets_to_file(self, filename):
        """ID 집합을 파일로 저장 - 모든 ID 집합 포함"""
        # 모든 탐지된 ID가 집합에 포함되도록 보장
        self._ensure_all_ids_in_sets()
        
        # 최종 ID 집합 생성 (대표 ID를 키로 사용)
        final_id_sets = {}
        
        # 비어있지 않은 집합들만 처리
        for set_idx, id_set in enumerate(self.id_sets):
            if not id_set:  # 빈 집합은 무시
                continue
                
            # 대표 ID 가져오기
            display_id = self.set_to_display_id.get(set_idx, min(id_set))
            
            # 최종 결과에 추가
            if display_id in final_id_sets:
                # 이미 존재하는 대표 ID면 집합 합치기
                final_id_sets[display_id].update(id_set)
            else:
                # 새 대표 ID면 새로운 집합 생성
                final_id_sets[display_id] = set(id_set)
        
        # 파일에 결과 저장
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # 대표 ID 순서대로 정렬해서 저장
                for display_id in sorted(final_id_sets.keys()):
                    id_set = final_id_sets[display_id]
                    # 모든 ID 집합 저장 (크기 제한 없음)
                    id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                    f.write(f"{display_id}: {id_set_str}\n")
            
            print(f"ID 집합이 성공적으로 저장되었습니다: {filename}")
            return True
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return False

def main():
    # 현재 스크립트 파일의 경로 가져오기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 현재 날짜와 시간으로 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 결과 저장 폴더 구조 생성
    results_dir = os.path.join(script_dir, "results")
    video_dir = os.path.join(results_dir, "video")
    txt_dir = os.path.join(results_dir, "txt")
    
    # 폴더가 없으면 생성
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    
    # 결과 파일 경로 설정
    txt_file = os.path.join(txt_dir, f"tracking_results_{timestamp}.txt")
    video_file = os.path.join(video_dir, f"tracking_video_{timestamp}.mp4")
    
    print(f"텍스트 결과 저장 경로: {txt_file}")
    print(f"영상 저장 경로: {video_file}")
    
    # 파일 경로 설정
    video_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\datas\0_8_IPC1_20230108162038.mp4"
    model_path = r"C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best.pt"
    
    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"파일이 존재하지 않습니다: {video_path}")
        return
    
    # YOLO 모델 로드
    try:
        model = YOLO(model_path)
        print("YOLO 모델 로드 성공")
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        return
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return
    
    # 비디오 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"원본 비디오 FPS: {fps}")
    
    # 저장 비디오 FPS 설정 (낮은 값으로 설정하여 영상 재생 시간 연장)
    output_fps = 10.0  # 10 FPS로 저장 (원본보다 낮게 설정)
    print(f"저장 비디오 FPS: {output_fps}")
    
    # 비디오 창 생성 (크기 조절 가능)
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    
    # ID 관리자 초기화
    id_manager = IDManager()
    
    # 탐지 설정
    conf_threshold = 0.7
    
    # 시간 제한 설정 (1분)
    start_time = time.time()
    duration = 1200  # 1200초 (20분)
    
    # 프레임 카운터 및 스킵 설정
    frame_count = 0
    processed_frames = 0
    frame_skip = 10  # 30프레임마다 1프레임만 처리
    
    # 관심 영역 설정 (필요에 따라 수정)
    roi_y1, roi_y2 = 800, 1600  # 세로 범위
    roi_x1, roi_x2 = 700, 1800   # 가로 범위
    
    # 첫 프레임을 읽어 관심 영역의 크기를 확인
    ret, first_frame = cap.read()
    if not ret:
        print("비디오 첫 프레임을 읽을 수 없습니다.")
        return
    
    # 관심 영역 설정 및 크기 확인
    roi_frame = first_frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_height, roi_width = roi_frame.shape[:2]
    
    # VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 코덱
    out = cv2.VideoWriter(video_file, fourcc, output_fps, (roi_width, roi_height))
    
    # 비디오 시작 위치를 다시 처음으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 총 프레임 수 확인
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"총 프레임 수: {total_frames}")
    
    # 처리된 프레임 저장용 리스트
    saved_frames = []
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("비디오 끝")
            break

        # 프레임 카운터 증가
        frame_count += 1
        
        # 프레임 스킵 (처리 속도 향상)
        if frame_count % frame_skip != 0:
            continue
        
        # 관심 영역 설정
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        
        # 윈도우 크기 자동 조정 (관심 영역 크기에 맞게)
        height, width = roi_frame.shape[:2]
        cv2.resizeWindow("Object Detection", width, height)
        
        # 처리된 프레임 수 증가
        processed_frames += 1
        
        # 현재 시간 확인
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # 진행 상황 표시 (100프레임마다)
        if processed_frames % 10 == 0:
            print(f"처리 중... 프레임: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%), 처리된 프레임: {processed_frames}")
        
        # 1분 경과 확인
        if elapsed_time >= duration:
            print(f"설정한 시간({duration}초) 경과. 처리 종료.")
            break
        
        # 화면에 시간 정보 표시용
        remaining_time = duration - elapsed_time

        # 클래스 인덱스 확인 (이 부분은 모델 로드 후 한 번만 실행)
        class_names = model.names
        # print("사용 가능한 클래스:", class_names)

        # chicks의 클래스 인덱스 찾기
        chicks_idx = None
        for idx, name in class_names.items():
            if 'chick' in name.lower():
                chicks_idx = idx
                # print(f"병아리(chicks) 클래스 인덱스: {chicks_idx}")
                break

        if chicks_idx is None:
            # print("경고: 'chicks' 클래스를 찾을 수 없습니다.")
            # 클래스명이 정확히 매칭되지 않을 경우 수동으로 인덱스 설정
            chicks_idx = 0  # 대부분의 경우 클래스 인덱스가 0부터 시작
        
        # 객체 탐지 (YOLOv8 추적 모드 사용)
        results = model.track(roi_frame, conf=conf_threshold, persist=True, classes=[chicks_idx])[0]
        
        # 결과 추출
        boxes = results.boxes.xyxy.cpu().tolist() if hasattr(results.boxes, 'xyxy') else []
        ids = results.boxes.id.int().cpu().tolist() if hasattr(results.boxes, 'id') else [-1] * len(boxes)
        
        # 탐지된 객체 정보
        detected_objects = [(box[0], box[1], box[2], box[3], id) for box, id in zip(boxes, ids)]
        
        # ID 관리자 업데이트
        id_manager.update(detected_objects)
        
        # 결과 시각화
        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = [int(p) for p in box]
            
            # 대표 ID 얻기
            if obj_id != -1:
                display_id = id_manager.get_display_id(obj_id)
                id_set = id_manager.get_id_set(obj_id)
                
                # ID 집합 크기에 따라 바운딩 박스 색상 결정
                if len(id_set) > 1:
                    # ID가 재할당된 객체는 빨간색으로 표시
                    box_color = (0, 0, 255)  # 빨간색(BGR)
                else:
                    # 일반 객체는 녹색
                    box_color = (0, 255, 0)  # 녹색(BGR)
                
                # 바운딩 박스 그리기
                cv2.rectangle(roi_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # ID 집합이 2개 이상인 경우에만 레이블 표시
                if len(id_set) > 1:
                    # 집합 형식으로 표시 {고유id, 새로 할당된 id...}
                    id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
                    
                    # 배경 박스 그리기 (텍스트 가독성 향상)
                    (text_width, text_height), _ = cv2.getTextSize(id_set_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(roi_frame, 
                                 (int(x1), int(y1) - text_height - 5), 
                                 (int(x1) + text_width + 5, int(y1)), 
                                 (0, 0, 150), -1)  # 어두운 빨간색 배경
                    
                    # 텍스트 표시
                    cv2.putText(roi_frame, id_set_str, (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 시간 정보 표시
        cv2.putText(roi_frame, f"time: {int(remaining_time)}s", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 프레임 번호 표시
        cv2.putText(roi_frame, f"frame: {frame_count}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                  
        # 영상 파일에 프레임 저장
        out.write(roi_frame)
        saved_frames.append(roi_frame)
        
        # 화면에 표시
        cv2.imshow("Object Detection", roi_frame)
        
        # 키 입력 처리 (q: 종료)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자가 종료했습니다.")
            break
    
    # 결과 저장
    print(f"ID 추적 결과를 저장합니다: {txt_file}")
    id_manager.save_id_sets_to_file(txt_file)
    
    # 저장된 영상 정보 출력
    print(f"저장된 프레임 수: {len(saved_frames)}")
    print(f"저장 영상 길이(예상): {len(saved_frames)/output_fps:.2f}초")
    
    # 자원 해제
    cap.release()
    out.release()  # VideoWriter 자원 해제
    cv2.destroyAllWindows()
    
    print(f"영상 저장이 완료되었습니다: {video_file}")
    print(f"실제 처리된 프레임: {processed_frames}/{frame_count} (전체 프레임의 {processed_frames/frame_count*100:.1f}%)")

if __name__ == "__main__":
    main()