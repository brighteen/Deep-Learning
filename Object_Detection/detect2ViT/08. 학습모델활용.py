import torch
import timm
from torchvision import transforms
from PIL import Image
import cv2

from ultralytics.nn.tasks import DetectionModel

# 안전하게 체크포인트 로드 (신뢰할 수 있는 출처라면)
with torch.serialization.safe_globals([DetectionModel]):
    checkpoint = torch.load("Object_Detection/객체탐지 실습/best_chick.pt",
                              map_location=torch.device("cpu"))
    # 체크포인트에서 모델 가중치만 추출
    state_dict = checkpoint["model"].model.state_dict()

# 1. 모델 생성 (원래 학습에 사용한 모델 아키텍처와 동일해야 함)
student_model = timm.create_model('deit_tiny_patch16_224', pretrained=False)
# 2. 모델 가중치 로드 (strict=False 옵션을 사용해 불일치하는 키 무시)
student_model.load_state_dict(state_dict, strict=False)
student_model.eval()

# 3. 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def is_dead_chicken(region_img):
    # BGR -> RGB 변환 후 PIL 이미지로 전환
    img = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = preprocess(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        output = student_model(input_tensor)
    prob = torch.softmax(output, dim=1)
    print("[debug] 모델 출력 확률:", prob)
    
    # 예: '닭'에 해당하는 클래스가 인덱스 1이라고 가정
    return prob[0, 1].item() > 0.5

# 영상 처리 코드
file_path = "Object_Detection/객체탐지 실습/datas/tile_r0_c3.mp4"
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
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            region = frame[y:y+h, x:x+w]
            if is_dead_chicken(region):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Dead Chicken", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow("Detection", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
