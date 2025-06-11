import pickle
import numpy as np
# 파일 경로
path = './data/avatars/wav2lip512_taeri/coords.pkl'
new_path = './data/avatars/wav2lip512_taeri/resized_coords.pkl'
# path = './data/avatars/wav2lip512_taeri/0512.pkl'

# 기존 pkl 불러오기
with open(path, 'rb') as f:
    coords = pickle.load(f)

print(f"총 {len(coords)}개의 좌표를 불러왔습니다.")

# 모든 좌표를 절반으로 줄이기 (예: (x1, y1, x2, y2) → (x1/2, y1/2, x2/2, y2/2))
scaled_coords = []
for i, box in enumerate(coords):
    if isinstance(box, (tuple, list)) and len(box) == 4:
        scaled_box = tuple([np.int64(v / 2) for v in box])
        scaled_coords.append(scaled_box)
    else:
        print(f"[경고] {i}번째 항목은 4개의 좌표값이 아닙니다: {box}")
        scaled_coords.append(box)  # 그대로 저장할지 무시할지는 선택

# 덮어쓰기 저장 (백업이 필요하면 다른 경로 지정)
with open(new_path, 'wb') as f:
    pickle.dump(scaled_coords, f)

print(f"좌표를 절반으로 줄인 후 다시 저장 완료: {new_path}")
