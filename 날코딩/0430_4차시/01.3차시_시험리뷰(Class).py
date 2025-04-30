class Counter:
    def __init__(self):
        self.count = 0 # count 변수 0으로 초기화

    def increment(self):
        # print(f'[degub] count : {self.count}')
        self.count += 1
        # print(f'[debug] count : {self.count}')
        # return self.count
    
    def reset(self):
        self.count = 0
        # return self.count

    def get(self):
        return self.count

c = Counter()
print(f'count init: {c.count}') # 0
c.increment() # 1
print(f'count increment: {c.count}') # 1
c.reset() # 0
print(f'count reset: {c.count}') # 0
print(f'count get: {c.get()}')