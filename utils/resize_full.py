import os
from PIL import Image

# 입력 폴더 경로
input_folder = "./data/avatars/wav2lip512_taeri/full_imgs"

# 폴더 내 모든 .jpg 파일 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):
        input_path = os.path.join(input_folder, filename)

        # 이미지 열기
        img = Image.open(input_path)
        width, height = img.size

        # 가로/세로 절반 크기로 리사이즈
        new_size = (width, height)
        img_resized = img.resize(new_size, Image.LANCZOS)

        # 저장할 PNG 파일 경로
        new_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(input_folder, new_filename)

        # PNG 저장
        img_resized.save(output_path, format='PNG')
        print(f"Saved resized PNG: {new_filename} ({new_size})")

        # 원본 JPG 삭제
        os.remove(input_path)
        print(f"Deleted original JPG: {filename}")
