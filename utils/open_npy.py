import numpy as np

# 파일 경로
path = './data/avatars/wav2lip512_taeri/0512.npy'

# .npy 파일 열기
coords = np.load(path, allow_pickle=True)  # allow_pickle=True는 객체 저장된 경우 필요

# 전체 갯수 확인 (예: 550개)
print(f"총 {len(coords)}개의 좌표 정보가 있습니다.")

# 전체 타입 확인
print("전체 객체 타입:", type(coords))

# 예: 첫 번째 항목 확인
sample_index = 0
print(f"{sample_index}번째 항목 타입:", type(coords[sample_index]))
print(f"{sample_index}번째 항목 내용:", coords[sample_index])
