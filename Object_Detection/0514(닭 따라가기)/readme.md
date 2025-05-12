# 위치 기반 ID 연결 데이터 구조 방법

위치 기반으로 닭의 ID를 연결하기 위한 여러 데이터 구조 표현법을 소개해 드리겠습니다.

## 1. 딕셔너리(Dictionary) 표현

```python
# 1. ID 매핑 딕셔너리
id_mapping = {
    2: [2, 11, 35],  # 원본 ID를 키로, 모든 연결 ID를 리스트로
}

# 2. ID 변환 기록 딕셔너리
id_transitions = {
    2: 11,   # 이전 ID: 새 ID
    11: 35
}

# 3. 정규화된 ID 딕셔너리 (현재 ID를 원본 ID로 매핑)
canonical_id = {
    2: 2,
    11: 2,
    35: 2
}
```

## 2. 리스트(List) 표현

```python
# 1. 같은 닭의 ID 시퀀스
chicken_tracks = [
    [2, 11, 35],  # 첫 번째 닭의 모든 ID
    # [4, 8, 16]  # 다른 닭들의 ID 시퀀스
]

# 2. ID 전환 이력 리스트
id_changes = [
    (2, 11),  # (이전 ID, 새 ID)
    (11, 35)
]
```

## 3. 집합(Set) 표현

```python
# 같은 닭으로 간주되는 ID들의 집합
same_chicken = {2, 11, 35}

# 여러 닭들의 ID 집합 리스트
all_chickens = [
    {2, 11, 35},
    # {4, 8, 16}
]
```

## 4. 그래프(Graph) 표현

```python
# 인접 리스트로 표현한 ID 전환 그래프
id_graph = {
    2: [11],
    11: [35],
    35: []
}

# 네트워크 라이브러리 사용 (NetworkX)
import networkx as nx
G = nx.DiGraph()
G.add_edges_from([(2, 11), (11, 35)])
```

## 5. Union-Find (서로소 집합) 구조

```python
class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        self.parent[self.find(y)] = self.find(x)

# 사용 예
uf = UnionFind()
uf.union(2, 11)
uf.union(11, 35)
# uf.find(35)는 2를 반환 (모두 같은 닭으로 연결됨)
```

## 6. 클래스(Class) 기반 표현

```python
class Chicken:
    def __init__(self, original_id):
        self.original_id = original_id
        self.all_ids = [original_id]
        self.current_id = original_id
        self.positions = {}  # {프레임: (x, y, width, height)}
    
    def update_id(self, new_id, position, frame_num):
        self.all_ids.append(new_id)
        self.current_id = new_id
        self.positions[frame_num] = position

# 닭 객체들을 관리하는 클래스
class ChickenTracker:
    def __init__(self):
        self.chickens = {}  # 원본 ID: Chicken 객체
        self.id_to_original = {}  # 현재 ID: 원본 ID
    
    def register_id_change(self, old_id, new_id, position, frame_num):
        # ID 변경 등록 로직
        pass
```

이 중에서 구현의 복잡성과 성능을 고려할 때, 딕셔너리 방식의 `canonical_id`나 `UnionFind` 구조가 효율적일 수 있습니다. 필요에 따라 위치 정보와 시간 정보를 추가로 저장하면 더욱 강건한 추적이 가능합니다.