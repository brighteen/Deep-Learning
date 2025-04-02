import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import timm

# (1) 모델 로드 및 전처리 설정
student_model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
student_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# (2) 분류 함수: 후보 영역을 받아 softmax 결과를 출력하고, 임계값 초과 시 True 반환
def is_dead_chicken(region_img):
    # BGR 이미지를 RGB로 변환 후 PIL 이미지로 전환
    img = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = preprocess(pil_img).unsqueeze(0) # 배치 차원 추가
    
    with torch.no_grad():
        output = student_model(input_tensor)
    prob = torch.softmax(output, dim=1)
    # 디버깅용: 전체 softmax 확률 출력
    print("[debug] 모델 출력 확률:", prob)
    
    # 예를 들어, 닭에 해당하는 클래스 인덱스가 1이라고 가정할 때
    return prob[0, 1].item() > 0.5


# (3) 영상 읽기 및 프레임 처리
file_path = "Object_Detection/객체탐지 실습/datas/tile_r0_c0.mp4"  # 영상 파일 경로
cap = cv2.VideoCapture(file_path)

ret, prev_frame = cap.read()
if not ret:
    print("비디오를 읽어오지 못했습니다.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("더 이상 프레임을 읽을 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 프레임 차분 계산
    frame_diff = cv2.absdiff(prev_gray, gray)
    # 차분 결과 이진화
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    # 팽창 연산으로 노이즈 제거
    dilated = cv2.dilate(thresh, None, iterations=2)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 임시로 면적 50 이상인 모든 영역을 대상으로 처리 (필요에 따라 조건 조정)
        if area > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            region = frame[y:y+h, x:x+w]
            # 분류 함수 결과 확인 및 디버깅: 모델 출력 확률이 콘솔에 출력됨
            if is_dead_chicken(region):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Dead Chicken", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("Classification Check", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
