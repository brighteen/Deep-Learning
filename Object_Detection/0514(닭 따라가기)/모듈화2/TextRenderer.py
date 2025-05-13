from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

class TextRenderer:
    """한글 텍스트를 이미지에 렌더링하는 클래스"""
    
    @staticmethod
    def put_text(img, text, position, font_size=30, font_color=(0, 255, 0), font_thickness=2, with_background=False, bg_color=(0, 0, 0), bg_opacity=0.7):
        """
        한글을 포함한 텍스트를 이미지에 그립니다.
        
        Args:
            img (numpy.ndarray): 이미지 배열
            text (str): 표시할 텍스트
            position (tuple): 텍스트를 표시할 위치 (x, y)
            font_size (int): 폰트 크기
            font_color (tuple): 폰트 색상 (B, G, R)
            font_thickness (int): 폰트 두께
            with_background (bool): 배경을 그릴지 여부
            bg_color (tuple): 배경 색상 (B, G, R)
            bg_opacity (float): 배경 불투명도 (0.0-1.0)
            
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
          # 배경 그리기 (OpenCV로 처리)
        if with_background:
            img_cv = np.array(img_pil)
            # 텍스트 크기 측정 (PIL 2.0+ 호환성 유지)
            # PIL 8.0.0부터 textsize 대신 textbbox 또는 textlength 사용
            try:
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                text_width = right - left
                text_height = bottom - top
            except AttributeError:
                # 구버전 PIL에서는 textsize 사용
                text_width, text_height = draw.textsize(text, font=font)
            
            # 배경 직사각형 계산 (상하좌우 여백 추가)
            padding = 4
            x, y = position
            x1, y1 = x - padding, y - padding - int(font_size * 0.8)  # 상단 여백 조정
            x2, y2 = x + text_width + padding, y + padding
            
            # 배경 직사각형 그리기
            overlay = img_cv.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
            # 투명도 적용
            img_cv = cv2.addWeighted(overlay, bg_opacity, img_cv, 1 - bg_opacity, 0)
            
            # 다시 PIL 이미지로 변환
            img_pil = Image.fromarray(img_cv)
            draw = ImageDraw.Draw(img_pil)
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=font_color[::-1])  # RGB -> BGR 변환을 위해 color를 뒤집음
        
        # PIL 이미지를 NumPy 배열로 변환하여 반환
        return np.array(img_pil)
