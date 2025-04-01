import json
import os

def convert_bbox(size, box):
    """
    COCO bbox: [x_min, y_min, width, height]
    YOLO bbox: [x_center, y_center, width, height] (모두 0~1 사이의 정규화 값)
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    return (x * dw, y * dh, w * dw, h * dh)

def coco_to_yolo(coco_json_path, output_dir):
    """
    coco_json_path: COCO 형식 어노테이션 파일 경로
    output_dir: YOLO 텍스트 파일을 저장할 폴더 (이미지와 동일한 이름으로 .txt 생성)
    """
    # JSON 파일 읽기
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    # 이미지 정보와 어노테이션을 이미지 ID별로 정리
    images_info = {image['id']: image for image in data['images']}
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)
    
    # 각 이미지에 대해 YOLO 형식의 라벨 파일 생성
    for img_id, image in images_info.items():
        file_name = image['file_name']
        width = image['width']
        height = image['height']
        # YOLO 형식의 라벨 파일 이름 (예: img1.jpg -> img1.txt)
        txt_filename = os.path.splitext(file_name)[0] + '.txt'
        txt_file_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_file_path, 'w') as out_file:
            # 해당 이미지에 어노테이션이 있다면 변환 수행
            if img_id in annotations:
                for ann in annotations[img_id]:
                    # 여기서는 모든 어노테이션을 chicken(클래스 id 0)으로 처리합니다.
                    bbox = ann['bbox']  # [x, y, width, height] (COCO 형식)
                    bbox_yolo = convert_bbox((width, height), bbox)
                    # 소수점 자릿수 조절이 필요하면 format 함수를 활용할 수 있습니다.
                    out_file.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        bbox_yolo[0], bbox_yolo[1], bbox_yolo[2], bbox_yolo[3]
                    ))

if __name__ == "__main__":
    coco_json_path = "_annotations.coco.json"  # COCO JSON 파일 경로
    output_dir = "dataset/labels"  # 라벨 파일을 저장할 폴더 (예: dataset/train 또는 dataset/val 하위 폴더)
    
    # output_dir 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    coco_to_yolo(coco_json_path, output_dir)
