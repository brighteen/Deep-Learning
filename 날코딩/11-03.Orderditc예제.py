from collections import OrderedDict
# value1 정의
value1 = {'name' : 'Charlotte', 'age' : 27, 'city' : 'Busan'}
print(f"\nvalue: {value1}, shape: {type(value1)}")
print(f"length of value1: {len(value1)}")

# value2 정의
value2 = {'name' : 'fruit', 'age' : 3, 'city' : 'Seoul'}
print(f"\nvalue: {value2}, shape: {type(value2)}")
print(f"length of value2: {len(value2)}")

# 빈 dict생성
d = {}
print(f"\ndict: {d}, shape: {type(d)}")

# dict에 value1 추가``
d["key"] = value1
print(f"\ndict에 value1 추가: \n{d}, shape: {type(d)}")
print(f"\nd['key']: {d['key']}")

d["key2"] = value2  # dict에 value2 추가
print(f"\ndict에 value2 추가: \n{d}, shape: {type(d)}")
print(f"\nd['key2']: {d['key2']}")  # d['key2'] 출력

# 
# print(f"key의 name의 밸류값 출력: {d["key"]["name"]}") # SyntaxError: f-string: unmatched '['

print(f"key의 age의 밸류값 출력: {d['key']['age']}") # 'Charlotte' 출력

print(d.get("key"))  # 'key'의 밸류값 출력



# od = OrderedDict()
# od["key"] = value1
# print(f"\nOrderedDict: {od}, shape: {type(od)}")