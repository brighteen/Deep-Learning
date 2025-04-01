import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# (1) 딥러닝 모델 로드
# 미리 학습된 DeiT-Tiny 모델을 불러온다고 가정합니다.
# 실제 모델 파일 경로와 클래스 구조에 맞게 수정해야 합니다.
import timm
# student_model = torch.load("deit_tiny_model.pth", map_location=torch.device('cpu'))
student_model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
student_model.eval()

# 모델 입력 전처리: 일반적으로 ImageNet 기준으로 학습된 모델 전처리 사용
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def is_dead_chicken(region_img):
    """
    후보 영역(region_img)을 받아서, 해당 영역이 닭인지(또는 죽은 닭일 가능성이 높은지)를 분류합니다.
    """
    # BGR -> RGB 변환 후 전처리
    img = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = preprocess(pil_img).unsqueeze(0)  # 배치 차원 추가

    with torch.no_grad():
        output = student_model(input_tensor)
    # 소프트맥스 적용 후, 특정 클래스(예: index 1: 닭)의 확률이 임계값(예: 0.5) 이상이면 True 반환
    prob = torch.softmax(output, dim=1)
    if prob[0, 1] > 0.5:
        return True
    return False
# (2) 영상에서 정지 객체 검출 및 후보 영역 추출
file_path = "3_1/detect/폐사체동영상/0_8_IPC1_20220912080906.mp4"  # 분석할 영상 파일 경로
cap = cv2.VideoCapture(file_path)  # 분석할 영상 파일 경로

ret, prev_frame = cap.read()
if not ret:
    print("비디오를 읽어오지 못했습니다.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 프레임 차분으로 움직임 계산
    frame_diff = cv2.absdiff(prev_gray, gray)
    # 임계값 적용 및 이진화
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    # 팽창 연산을 통해 잡음 제거
    dilated = cv2.dilate(thresh, None, iterations=2)
    # 윤곽선 추출
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 닭의 예상 크기에 따른 영역 필터링 (예: 200 ~ 800 픽셀)
        if 200 < area < 800:
            x, y, w, h = cv2.boundingRect(cnt)
            region = frame[y:y+h, x:x+w]
            # 후보 영역 분류: 정지 객체가 닭으로 확인되면 'Dead Chicken' 표시
            if is_dead_chicken(region):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Dead Chicken", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Dead Chicken Detection", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    prev_gray = gray  # 다음 비교를 위해 현재 프레임을 저장

cap.release()
cv2.destroyAllWindows()
