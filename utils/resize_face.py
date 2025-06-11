import os
from PIL import Image

# 원본 jpg 이미지 폴더
input_folder = "./data/avatars/wav2lip512_taeri/face_imgs"

# 리사이즈 크기
target_size = (512, 512)

# 폴더 내 모든 .jpg 파일 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):
        input_path = os.path.join(input_folder, filename)

        # 이미지 열기
        img = Image.open(input_path)

        # 리사이즈
        img_resized = img.resize(target_size, Image.LANCZOS)

        # 새 파일명 (확장자 .png로 변경)
        new_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(input_folder, new_filename)

        # PNG로 저장
        img_resized.save(output_path, format='PNG')

        print(f"Converted and resized: {filename} → {new_filename}")

        # 원본 JPG 삭제
        os.remove(input_path)
        print(f"Deleted original JPG: {filename}")
