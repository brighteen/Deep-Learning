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
