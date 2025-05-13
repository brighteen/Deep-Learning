class ChickenDetector:
    """YOLO 모델을 사용하여 닭을 감지하는 클래스"""
    
    def __init__(self, model_path):
        """
        YOLO 모델을 로드합니다.
        
        Args:
            model_path (str): YOLO 모델 파일 경로
        """
        try:
            self.model = YOLO(model_path)
            self.enabled = True
            print(f"YOLO 모델을 성공적으로 로드했습니다: {model_path}")
        except Exception as e:
            print(f"YOLO 모델 로드 실패: {e}")
            self.enabled = False
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
            
        # YOLOv8 모델로 탐지
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)[0]
        
        # 결과에서 닭 개체 수 계산
        boxes = results.boxes
        chicken_count = len(boxes)
        
        # 결과와 닭 개수 반환
        return results, chicken_count
