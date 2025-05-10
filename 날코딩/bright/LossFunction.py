class MeanSquaredError:
    def __init__(self):
        self.y1 = None
        self.y2 = None
        self.t = None
        self.loss = None

    def forward(self, y2, t):
        self.y2 = y2
        self.t = t
        self.loss = 0.5 * (self.y2 - self.t)**2
        return self.loss
    
    def backward(self, dout=1):
        return (self.y2 - self.t) * dout