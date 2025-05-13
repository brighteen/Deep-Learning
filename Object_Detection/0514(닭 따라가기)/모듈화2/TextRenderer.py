class TextRenderer:
    """한글 텍스트를 이미지에 렌더링하는 클래스"""
    
    @staticmethod
    def put_text(img, text, position, font_size=30, font_color=(0, 255, 0), font_thickness=2):
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
