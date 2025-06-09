import numpy as np
from collections import OrderedDict

from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss

class SimpleConvNet:
    """단순한 합성곱 신경망
    
    conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size : 입력 크기 (MNIST의 경우엔 784)
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트 (e.g. [100, 100, 100])
    output_size : 출력 크기 (MNIST의 경우엔 10)
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정 (e.g. 0.01)
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':3, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]


    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

if __name__ == '__main__':
    '''
    신경망 구조:
    Conv - Relu - Pool - Affine - Relu - Affine - SoftmaxWithLoss
    입력 크기: (1, 5, 5)
    합성곱층: 필터 수 2, 필터 크기 3, 패딩 0, 스트라이드 1
    은닉층: 3 뉴런, 출력층: 2 뉴런
    가중치 초기화 표준편차: 0.01
    '''
    net = SimpleConvNet(input_dim=(1, 4, 4), 
                        conv_param={'filter_num':2, 'filter_size':3, 'pad':0, 'stride':1},
                        hidden_size=3, output_size=2, weight_init_std=0.01)
    
    print(f"\n각 레이어 정보: {net.layers.keys()}")

    print(f"레이어 타입: {type(net.layers)}")
    print(f"손실함수 타입: {type(net.last_layer)}")
    print(f"파라미터 타입: {type(net.params)}")

    print(f"손실함수: {net.last_layer.__class__.__name__}")

    print(f"\n가중치 매개변수 정보: {net.params.keys()}")
    print(f"W1 shape: {np.shape(net.params['W1'])}")
    print(f"b1 shape: {np.shape(net.params['b1'])}")
    print(f"W2 shape: {np.shape(net.params['W2'])}")
    print(f"b2 shape: {np.shape(net.params['b2'])}")
    print(f"W3 shape: {np.shape(net.params['W3'])}")
    print(f"b3 shape: {np.shape(net.params['b3'])}")

    
    # x = np.random.rand(100, 1, 28, 28)  # 임의의 입력 데이터
    # t = np.random.randint(0, 10, size=(100,))  # 임의의 정답 레이블
    
    # loss = net.loss(x, t)
    # print("Loss:", loss)
    
    # grads = net.gradient(x, t)
    # print("Gradients:", grads.keys())