import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"

    # 좌표 불러오기
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)

    return full_imgs_path, face_imgs_path, coord_list_cycle

# 테스트용 아바타 ID
avatar_id = "wav2lip512_taeri"
full_imgs_path, face_imgs_path, coord_list_cycle = load_avatar(avatar_id)

# 예시 인덱스 선택
idx = 0

# full 이미지 로드
full_img = cv2.imread(os.path.join(full_imgs_path, f"{idx:05d}.png"))
full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

# 입 프레임 (여기선 face image를 res_frame 역할로 가정)
res_frame = cv2.imread(os.path.join(face_imgs_path, f"{idx:05d}.png"))
res_frame = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)

# deep copy 후 bbox에 입 덮어쓰기
combine_frame = copy.deepcopy(full_img)

# bbox 불러오기: (y1, y2, x1, x2)
y1, y2, x1, x2 = coord_list_cycle[idx]

# 합성된 얼굴 붙이기
try:
    resized_res_frame = cv2.resize(res_frame, (x2 - x1, y2 - y1))
    combine_frame[y1:y2, x1:x2] = resized_res_frame

    # 🔴 붉은색 사각형 표시
    cv2.rectangle(combine_frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

except Exception as e:
    print(f"[ERROR] Resize or paste failed: {e}")


# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Full Image")
plt.imshow(full_img)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Face Frame (res_frame)")
plt.imshow(res_frame)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Combined Image")
plt.imshow(combine_frame)
plt.axis('off')

plt.tight_layout()
plt.show()
