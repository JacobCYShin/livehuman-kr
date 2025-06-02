import pickle

# 파일 경로
path = './data/avatars/wav2lip256_avatar1/coords.pkl'

# .pkl 파일 열기
with open(path, 'rb') as f:
    coords = pickle.load(f)

# 전체 갯수 확인 (예: 550개)
print(f"총 {len(coords)}개의 좌표 정보가 있습니다.")

# 예: 첫 번째 이미지에 대한 좌표 정보 확인
sample_index = 0  # 확인하고 싶은 이미지 인덱스 (0부터 시작)
print(f"{sample_index}번째 이미지 좌표:", coords[sample_index])


# 전체 타입 확인
print("전체 객체 타입:", type(coords))

# 전체 길이 (예: 550개 좌표)
print("총 항목 수:", len(coords))

# 첫 번째 항목의 타입과 내용 보기
print("첫 번째 항목 타입:", type(coords[0]))
print("첫 번째 항목 내용:", coords[0])
