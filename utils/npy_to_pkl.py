import numpy as np
import pickle

# npy 파일 경로 (입력)
npy_path = './data/avatars/wav2lip512_taeri/0512.npy'

# pkl 파일 경로 (출력)
pkl_path = './data/avatars/wav2lip512_taeri/0512.pkl'

# npy 파일 로드
coords_np = np.load(npy_path, allow_pickle=True)

# (x1, y1, x2, y2) → (y1, y2, x1, x2) + np.int64로 명시적 캐스팅
coords_list = [
    (np.int64(y1), np.int64(y2), np.int64(x1), np.int64(x2))
    for (x1, y1, x2, y2) in coords_np
]

# pkl 파일로 저장
with open(pkl_path, 'wb') as f:
    pickle.dump(coords_list, f)

print(f"{len(coords_list)}개의 좌표를 .pkl 파일로 저장했습니다.")
