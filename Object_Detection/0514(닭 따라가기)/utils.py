import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

def put_text_on_image(img, text, position, font_size=30, font_color=(0, 255, 0), font_thickness=2):
    """
    한글을 포함한 텍스트를 이미지에 그립니다.
    
    Args:
        img (numpy.ndarray): 이미지 배열
        text (str): 표시할 텍스트
        position (tuple): 텍스트를 표시할 위치 (x, y)
        font_size (int): 폰트 크기
        font_color (tuple): 폰트 색상 (B, G, R)
        font_thickness (int): 폰트 두께
        
    Returns:
        numpy.ndarray: 텍스트가 추가된 이미지
    """
    # 이미지를 PIL 형식으로 변환
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # 폰트 로드 (Windows에서는 기본 폰트 "malgun.ttf"를 사용)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)  # 윈도우 기본 한글 폰트
    except:
        # 폰트를 찾을 수 없는 경우 기본 폰트 사용
        font = ImageFont.load_default()
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=font_color[::-1])  # RGB -> BGR 변환을 위해 color를 뒤집음
    
    # PIL 이미지를 NumPy 배열로 변환하여 반환
    return np.array(img_pil)

def create_grid_display(frame, grid_size, target_cell=None):
    """
    이미지에 그리드를 그리고 셀 번호를 표시합니다.
    
    Args:
        frame (numpy.ndarray): 원본 이미지
        grid_size (int): 그리드 크기
        target_cell (tuple): 강조할 대상 셀 (row, col)
        
    Returns:
        numpy.ndarray: 그리드가 그려진 이미지
    """
    display_frame = frame.copy()
    frame_height, frame_width = display_frame.shape[:2]
    
    # 그리드 셀 크기 계산
    cell_height = frame_height // grid_size
    cell_width = frame_width // grid_size
    
    # 가로선 그리기
    for i in range(1, grid_size):
        y = i * cell_height
        cv2.line(display_frame, (0, y), (frame_width, y), (0, 255, 0), 2)
    
    # 세로선 그리기
    for i in range(1, grid_size):
        x = i * cell_width
        cv2.line(display_frame, (x, 0), (x, frame_height), (0, 255, 0), 2)
    
    # 각 셀에 번호 표시
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell_width + cell_width // 2 - 20
            y = row * cell_height + cell_height // 2
            
            # 대상 셀 강조
            if target_cell and (row, col) == target_cell:
                cv2.putText(display_frame, f"({row},{col})", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                # 셀에 사각형 강조
                cv2.rectangle(display_frame, 
                            (col * cell_width, row * cell_height), 
                            ((col+1) * cell_width, (row+1) * cell_height), 
                            (0, 0, 255), 3)
            else:
                cv2.putText(display_frame, f"({row},{col})", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return display_frame

def get_cell_frame(frame, cell, grid_size):
    """
    전체 프레임에서 특정 셀 영역만 추출합니다.
    
    Args:
        frame (numpy.ndarray): 전체 프레임
        cell (tuple): 추출할 셀 좌표 (row, col)
        grid_size (int): 그리드 크기
        
    Returns:
        tuple: (y, x, cell_height, cell_width, cell_frame) - 셀 위치와 프레임
    """
    row, col = cell
    height, width = frame.shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size
    x = col * cell_width
    y = row * cell_height
    
    cell_frame = frame[y:y+cell_height, x:x+cell_width].copy()
    return y, x, cell_height, cell_width, cell_frame
