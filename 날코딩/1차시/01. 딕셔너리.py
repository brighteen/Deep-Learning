a = {'사과': '열매', '바나나': '열매', '당근': '채소', '양파': '채소'}
print(a)
print(type(a)) # <class 'dict'>
print(a.keys()) # dict_keys(['사과', '바나나', '당근', '양파'])
print(a.values()) # dict_values(['열매', '열매', '채소', '채소'])

print('---' * 20)

b = {1:2, 2:4, 3:6} # 정수형도 키로 사용 가능
print(b) # {1: 2, 2: 4, 3: 6}
print(type(b)) # <class 'dict'>
print(b.keys()) # dict_keys([1, 2, 3])
print(b.values()) # dict_values([2, 4, 6])

print('---' * 20)

c = {1.0 : '사과', 2.0 : '바나나', 3.0 : '당근'} # 실수형도 키로 사용 가능능
print(c) # {1.0: '사과', 2.0: '바나나', 3.0: '당근'}
print(type(c)) # <class 'dict'>
print(c.keys()) # dict_keys([1.0, 2.0, 3.0])
print(c.values()) # dict_values(['사과', '바나나', '당근'])

print('---' * 20)

print(type(1j)) # <class 'complex'> 복소수
print(1j) # 1j
print(1j * 1j) # -1+0j

print('---' * 20)


d = {True: '사과', False: '바나나'} # 불리언도 키로 사용 가능
print(d) # {True: '사과', False: '바나나'}
print(type(d)) # <class 'dict'>
print(d.keys()) # dict_keys([True, False])
print(d.values()) # dict_values(['사과', '바나나'])
