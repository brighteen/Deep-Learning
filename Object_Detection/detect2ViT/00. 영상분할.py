import cv2
import os

# 입력 영상 파일명 (필요에 따라 수정)
input_filename = '3_1/detect/폐사체동영상/0_8_IPC1_20220912080906.mp4'

# 저장 폴더 생성 (py 파일과 같은 위치)
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_my_data')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 영상 열기
cap = cv2.VideoCapture(input_filename)
if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

# 영상 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 앞 3분(180초)에 해당하는 프레임 수 계산
max_frames = int(fps * 180)

# 분할 grid 설정 (가로 5, 세로 4)
cols, rows = 5, 4
tile_width = width // cols
tile_height = height // rows

# VideoWriter 객체 20개 생성 (각 타일별)
writers = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
for r in range(rows):
    for c in range(cols):
        output_filename = os.path.join(save_dir, f'tile_r{r}_c{c}.mp4')
        writer = cv2.VideoWriter(output_filename, fourcc, fps, (tile_width, tile_height))
        writers.append(writer)

frame_count = 0
while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 각 타일에 대해 프레임을 crop하여 저장
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x = c * tile_width
            y = r * tile_height
            # 영상의 오른쪽 또는 아래쪽 부분은 원본 나누기에서 remainder가 있을 수 있으므로,
            # 마지막 타일에서는 실제 width/height를 고려할 수 있습니다.
            crop = frame[y:y+tile_height, x:x+tile_width]
            writers[idx].write(crop)
            idx += 1

    frame_count += 1

# 자원 해제
cap.release()
for writer in writers:
    writer.release()

print(f"처리된 프레임 수: {frame_count} - 영상이 '{save_dir}'에 저장되었습니다.")
