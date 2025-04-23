import numpy as np

class NonLinearModel:
    """비선형 변환 및 학습 모델 클래스"""

    def __init__(self, input_data, target, learning_rate=0.01):
        np.random.seed(1)
        self.x = input_data
        self.t = target
        self.lr = learning_rate
        self.w = np.random.randn(1)
        self.b = 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_diff(x):
        return (1.0 - NonLinearModel.sigmoid(x)) * NonLinearModel.sigmoid(x)

    @staticmethod
    def mean_squared_error(y, t):
        return (y - t) ** 2

    def train(self, epochs=1000):
        for i in range(1, epochs + 1):
            # 순전파
            y = self.x * self.w + self.b
            z = self.sigmoid(y)
            loss = self.mean_squared_error(z, self.t)

            if i % 100 == 0:
                print(f'[{i}번째] w: {self.w}, b: {self.b}, z: {z}, loss: {loss}')

            # 역전파
            dLdz = 2 * (z - self.t)
            dzdy = self.sigmoid_diff(y)

            dydw = self.x
            dydb = 1

            dLdw = dydw * dzdy * dLdz
            dLdb = dydb * dzdy * dLdz

            # 가중치 및 편향 업데이트
            self.w -= self.lr * dLdw
            self.b -= self.lr * dLdb

# NonLinearModel 클래스 사용 예제
if __name__ == "__main__":
    model = NonLinearModel(input_data=2, target=1, learning_rate=0.01)
    model.train(epochs=1000)