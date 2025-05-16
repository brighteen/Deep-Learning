ID를 관리하는 부분을 찾아드리겠습니다. 프로젝트에서 ID 관리는 주로 ChickenDetector.py 파일에서 이루어집니다. 

ID를 관리하는 주요 부분은 ChickenDetector.py 파일에 있습니다. 핵심적인 ID 관리 부분을 정리해드리겠습니다:

## ID 관리 핵심 부분

### 1. ID 관리를 위한 데이터 구조 (초기화 부분)

`ChickenDetector` 클래스의 `__init__` 메서드에서 ID 관리를 위한 주요 데이터 구조를 정의합니다:

```python
# 객체 추적을 위한 데이터 구조
self.chicken_id_sets = []  # 같은 닭으로 판단된 ID의 집합들
self.id_to_set_index = {}   # ID가 어느 집합에 속하는지 매핑
self.id_to_position = {}    # ID별 위치 정보
self.id_to_last_frame = {}  # ID별 마지막 등장 프레임
self.id_to_last_time = {}   # ID별 마지막 등장 시간
self.set_to_display_id = {} # 각 집합의 대표 ID
```

### 2. ID 일관성 관리 (핵심 알고리즘)

`_update_id_mapping` 메서드는 ID 일관성 관리의 핵심 부분입니다. 이 메서드는 다음과 같은 작업을 수행합니다:

1. **최근에 사라진 ID 식별**:
   ```python
   recently_disappeared_ids = {}
   for old_id, last_time in self.id_to_last_time.items():
       if old_id not in current_id_set:
           time_diff = current_time - last_time
           if time_diff < self.time_threshold:
               recently_disappeared_ids[old_id] = (self.id_to_position[old_id], time_diff)
   ```

2. **ID 매칭**:
   ```python
   # 거리 계산 (Euclidean 대신 제곱 거리를 사용하여 sqrt 연산 제거 - 추가 최적화)
   distance_squared = (center[0] - old_pos[0])**2 + (center[1] - old_pos[1])**2
   
   # 거리 제곱과 시간 조건 확인 (임계값도 제곱해서 비교)
   if distance_squared < (self.distance_threshold**2):
       # 거리와 시간을 가중치로 조합한 점수 계산 (가까울수록, 시간차가 적을수록 점수 높음)
       match_score = distance_squared + (time_diff * 10)  # 시간 가중치 추가
       potential_matches.append((match_score, i, chicken_id, old_id))
   ```

3. **ID 집합 업데이트**:
   ```python
   # 기존 ID가 속한 집합을 확인 (해시 테이블 사용으로 O(1) 접근)
   if old_id in self.id_to_set_index:
       set_index = self.id_to_set_index[old_id]
       self.chicken_id_sets[set_index].add(chicken_id)
       self.id_to_set_index[chicken_id] = set_index
   else:
       # 기존 ID가 집합에 없는 경우 - 새로운 집합 생성
       set_index = len(self.chicken_id_sets)
       self.chicken_id_sets.append({old_id, chicken_id})
       self.id_to_set_index[old_id] = set_index
       self.id_to_set_index[chicken_id] = set_index
       self.set_to_display_id[set_index] = min(chicken_id, old_id)  # 작은 ID를 대표로
   ```

### 3. ID 매핑 정보 적용

`_modify_results_with_mapped_ids` 메서드는 탐지 결과에 매핑된 ID 정보를 적용합니다:

```python
# 원본 ID 배열은 수정할 수 없으므로, 대신 렌더링 시 표시할 ID 정보를 저장
current_ids = results.boxes.id.cpu().numpy()
display_ids = []
id_sets = []  # 각 객체가 속한 ID 집합

# 빠른 조회를 위해 사전 계산된 캐시 사용
for chicken_id in current_ids:
    chicken_id = int(chicken_id)
    # 일관된 ID 매핑 적용 (캐시된 데이터 사용)
    if chicken_id in id_to_set_index_cache:
        set_index = id_to_set_index_cache[chicken_id]
        if set_index < len(chicken_id_sets_cache):  # 유효 범위 확인
            display_id = set_to_display_id_cache.get(set_index, chicken_id)
            display_ids.append(display_id)
            id_sets.append(chicken_id_sets_cache[set_index])
```

### 4. VideoPlayer에서의 ID 사용

VideoPlayer 클래스는 ChickenDetector에서 계산된 ID 매핑 정보를 사용하여 화면에 표시합니다:

```python
# VideoPlayer.py 파일에서
# ID가 속한 집합 가져오기
id_set = self._get_id_set_for_detection(results, i, original_id)
if id_set:
    # 간결한 ID 세트 표시 형식으로 수정
    id_set_str = "{" + ", ".join(map(str, sorted(id_set))) + "}"
    text = f"ID:{consistent_id} {id_set_str}"
    frame = self._draw_id_text_with_background(frame, text, x1, y1)
```

## 주요 ID 관리 메커니즘 정리

1. **초기화**: 처음에 `chicken_id_sets`, `id_to_set_index` 등의 데이터 구조 초기화

2. **ID 매핑 업데이트**:
   - 현재 프레임의 ID와 이전에 사라진 ID 간의 거리 계산
   - 거리 및 시간 기반으로 가장 유사한 매칭 찾기
   - 매칭된 ID들은 같은 집합에 추가됨

3. **ID 표시**:
   - 각 ID 집합의 대표 ID(일반적으로 가장 작은 ID)가 화면에 표시
   - ID 집합 전체 정보를 문자열로 변환하여 표시 (`{1, 3, 7}` 형식)

4. **데이터 정리**:
   - 오래된 ID 데이터는 주기적으로 정리하여 메모리 효율성 유지

이 시스템을 통해 닭이 화면에서 사라졌다가 다시 나타나도 일관된 ID로 추적할 수 있습니다.