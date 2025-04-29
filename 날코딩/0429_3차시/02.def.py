# print(input('입력')) # 함수(매개변수)

def add(x=0, y=10): # defalut x=0, y=10
    y = x + y
    return y

# print(f'함수 호출: {add(x=1,y=2)}') # x=1, y=2
# print(f'함수 호출: {add(2)}') # x=2, y=10

# print(type(add)) # <class 'function'>
# print(type(add(1,2))) # <class 'int'>

# class 선언
class 사람:
    def __init__(self, 이름):
        self.이름 = 이름

    def 인사(self):
        # return print(f'안녕하세요, {self}입니다.')
        print('안녕하세요')
        # print(f'안녕하세요, {self.이름}입니다.')
    
h1 = 사람(이름='여명구')
print(h1.이름) # 여명구
h1.인사()