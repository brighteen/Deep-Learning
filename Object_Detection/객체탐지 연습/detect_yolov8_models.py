from ultralytics import YOLO
import cv2

# 여러 모델을 리스트로 지정 (필요한 만큼 추가 가능)
model_paths = [
    "yolov8n.pt",  # nano
    "yolov8s.pt",  # small
    "yolov8m.pt",  # medium
    "yolov8l.pt",  # large
    "yolov8x.pt",  # extra large
]

image_path = "test.jpeg"  # 추론할 이미지 경로
results_dict = {}         # 모델별 결과 저장용

for mp in model_paths:
    print(f"-----\nRunning inference with {mp}")
    model = YOLO(mp)

    # 모델 추론
    result = model.predict(image_path, save=True, conf=0.3, project="detect", name="pridict")  # 필요하면 save=True 등 옵션 추가

    # 첫 번째(유일한) 결과 객체를 꺼냄
    res = result[0]
    # 시각화를 위해 이미지에 박스를 그린 버전 생성
    plotted_img = res.plot()

    # 결과를 딕셔너리에 저장 (필요하면 바로 imshow로 확인 가능)
    results_dict[mp] = plotted_img

    # 만약 바로 OpenCV 창으로 보고 싶다면:
    cv2.imshow(f"Result - {mp}", plotted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이제 results_dict에 각 모델별 시각화된 이미지가 저장됨
# 필요하다면 파일로 저장도 가능
