import torch
import cv2
import numpy as np
from ultralytics import YOLO

# 1. 학습시킨 best_chick.pt 모델을 불러온다.
try:
    model = YOLO(r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\best_chick.pt')
except FileNotFoundError:
    print("Error: best_chick.pt 모델 파일을 찾을 수 없습니다. 모델 파일 경로를 확인해주세요.")
    exit()

# 영상 파일 경로를 지정해주세요.
video_path = r'C:\Users\brigh\Documents\GitHub\Deep-Learning\Object_Detection\MotionHistoryImage(MHI)\tile_r0_c3.mp4'  # 실제 영상 파일 경로로 변경해주세요.
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: 영상을 열 수 없습니다. 영상 파일 경로를 확인해주세요.")
    exit()

# 영상의 프레임 속도와 크기를 얻습니다.
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 결과 영상을 저장할 VideoWriter 객체를 생성합니다.
output_path = 'output_chick_highlighted.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (mp4v, XVID 등)
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False) # 흑백 이진화이므로 isColor=False

# 클래스 이름을 'chicken'으로 설정합니다. (best_chick.pt 모델이 닭을 'chicken'으로 분류한다고 가정합니다.)
chicken_class_name = 'chicken'

# 영상을 프레임 단위로 읽어옵니다.
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 2. 영상 내 닭들을 분류한다.
    results = model.predict(frame, verbose=False)

    # 현재 프레임을 흑백 이미지로 변환합니다.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary_frame = np.zeros_like(gray_frame)

    # 분류된 닭들을 밝게, 나머지는 어둡게 이진화합니다.
    for result in results:
        if result.boxes is not None and result.names is not None:
            for box, cls_id in zip(result.boxes, result.boxes.cls):
                class_name = result.names[int(cls_id)]
                if class_name == chicken_class_name:
                    # 바운딩 박스 좌표를 얻습니다.
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                    # 닭 영역을 흰색(255)으로 설정합니다.
                    binary_frame[y_min:y_max, x_min:x_max] = 255

    # 결과 영상을 저장합니다.
    out.write(binary_frame)

    # (선택 사항) 화면에 결과 프레임을 보여줍니다.
    # cv2.imshow('Binarized Video', binary_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 영상 처리 완료 후 리소스를 해제합니다.
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"닭이 분류되어 밝게 표시된 이진화 영상이 '{output_path}'로 저장되었습니다.")