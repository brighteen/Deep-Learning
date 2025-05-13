from ultralytics import YOLO

def load_yolo_model(model_path):
    """
    YOLO 모델을 로드합니다.
    
    Args:
        model_path (str): YOLO 모델 파일 경로
    
    Returns:
        tuple: (model, success) - 모델 객체와 로드 성공 여부
    """
    try:
        model = YOLO(model_path)
        return model, True
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        return None, False

def detect_chickens(model, frame, conf_threshold=0.5):
    """
    YOLO 모델을 사용하여 주어진 프레임에서 닭을 탐지합니다.
    
    Args:
        model: YOLO 모델
        frame: 탐지할 프레임
        conf_threshold: 탐지 확신도 임계값
        
    Returns:
        tuple: (results, chicken_count) - 탐지 결과와 닭 개수
    """
    # YOLOv8 모델로 탐지
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    
    # 결과에서 닭 개체 수 계산
    boxes = results.boxes
    chicken_count = len(boxes)
    
    # 결과와 닭 개수 반환
    return results, chicken_count

def track_chickens(model, frame, conf_threshold=0.5):
    """
    YOLO 모델로 닭을 추적합니다.
    
    Args:
        model: YOLO 모델
        frame: 처리할 프레임
        conf_threshold: 탐지 확신도 임계값
        
    Returns:
        tuple: (results, chicken_count) - 추적 결과와 닭 개수
    """
    # YOLOv8 모델의 track 메소드 사용
    results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)[0]
    
    # 결과에서 닭 개체 수 계산
    boxes = results.boxes
    chicken_count = len(boxes)
    
    return results, chicken_count
