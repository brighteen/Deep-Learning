hungry = True
sleepy = False

print(type(hungry))
print(hungry)
print(not hungry)
print(hungry and sleepy)
print(hungry or sleepy)

if hungry: # 조건이 True면
    print("Hungry yaa")
else:
    print("ddd")

for i in [1,2,3]:
    print(f"{i}번째 반복")

# 함수 정의
def hello(object1): # object1이라는 인수 지정
    print("hello world " + object1 + "!")

hello("yaay")

print("---\n")

class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")
    
    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good bye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()

import numpy as np

x = np.array([[1,2], [3,4]])
print(f"\nx_shape: {x.shape}")
print(f"x/20: \n{x / 2.0}")

y = np.array([[3,0], [0,6]])
print(f"\nx + y: \n{x+y}")

# 브로드캐스트
print("\n브로드캐스트: 2x2에 스칼라(10)를 곱하면 10으로 이루어진 2x2행렬를 곱한 연산이 됨.")
print(f"x * 10: \n {x*10}")

y = np.array([10,20])
print(f"x*y: \n{x*y}")

print("\n반복문으로 행렬 원소 접근")
for row in x:
    print(f"x: {row}")

print(f"flatten: {x.flatten()}, shape: {x.flatten().shape}")