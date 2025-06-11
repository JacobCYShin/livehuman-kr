import pickle

file_path = 'data/avatars/wav2lip512_taeri/resized_coords.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("Top-level structure:", type(data))
print("Length:", len(data))

# 앞부분 3~4개만 출력
for i in range(4):
    print(f"Item {i}: {data[i]}")  # 또는 tuple 내부를 unpack하려면: x, y, w, h = data[i]
