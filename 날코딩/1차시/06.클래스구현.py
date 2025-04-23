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
