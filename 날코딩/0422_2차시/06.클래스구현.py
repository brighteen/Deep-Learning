class Man:
    """샘플 클래스"""

    def __init__(self, name):
        # 클래스 초기화 메서드, name 속성을 초기화합니다.
        self.name = name
        print("Initilized!")

    def hello(self):
        # 인사 메서드, name 속성을 사용하여 인사 메시지를 출력합니다.
        print("Hello " + self.name + "!")

    def goodbye(self):
        # 작별 인사 메서드, name 속성을 사용하여 작별 메시지를 출력합니다.
        print("Good-bye " + self.name + "!")

# Man 클래스의 인스턴스를 생성하고 메서드를 호출합니다.
m = Man("andy") # 인스턴스 생성
m.hello() # hello 메서드 호출
m.goodbye() # goodbye 메서드 호출

class Animal:
    """동물 클래스"""

    def __init__(self, species, sound):
        # 클래스 초기화 메서드, species와 sound 속성을 초기화합니다.
        self.species = species
        self.sound = sound
        print(f"\nA {self.species} has been created!")

    def describe(self):
        # 동물의 정보를 출력하는 메서드
        print(f"This is a {self.species}.")

    def make_sound(self):
        # 동물이 소리를 내는 메서드
        print(f"The {self.species} says '{self.sound}'!")

# Animal 클래스의 인스턴스를 생성하고 메서드를 호출합니다.
dog = Animal("dog", "woof") # 인스턴스 생성
dog.describe() # describe 메서드 호출
dog.make_sound() # make_sound 메서드 호출

cat = Animal("cat", "meow") # 또 다른 인스턴스 생성
cat.describe() # describe 메서드 호출
cat.make_sound() # make_sound 메서드 호출
