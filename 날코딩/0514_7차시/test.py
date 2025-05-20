import numpy as np

class LinearModel:
    """
    LinearTransfer 변환 레이어: 신경망에서 완전연결(fully-connected) 계층을 구현
    y = x*W + b 형태의 선형 변환을 수행
    """
    def __init__(self, W, b):
        """
        Affine 레이어 초기화
        
        매개변수:
            W: 가중치 행렬
            b: 편향 벡터
        """
        self.W = W  # 가중치 행렬 저장
        print("Initial W!: ", self.W)
        print("="*20)
        self.b = b  # 편향 벡터 저장
        print("Initial b!: ", self.b)
        self.x = None  # 순전파 시 입력 데이터를 저장할 변수
        self.original_x_shape = None  # 입력 데이터의 원래 형상 저장 (텐서 처리 위함)
        # 가중치와 편향 매개변수의 미분
        self.dW = None  # 가중치에 대한 기울기 저장 변수
        self.db = None  # 편향에 대한 기울기 저장 변수

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx

if __name__ == "__main__":
    np.random.seed(0)
    W = np.random.randn(2,1)
    b  = np.zeros(1)

    model = LinearModel(W, b) # 
    print("="*20)
    print([f"W: {model.W}, shape: {np.shape(model.W)}"])
    print([f"b: {model.b}, shape: {np.shape(model.b)}"])