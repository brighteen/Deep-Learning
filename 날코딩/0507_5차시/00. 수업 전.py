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

def hello(object1):
    print("hello world " + object1 + "!")

hello("yaay")