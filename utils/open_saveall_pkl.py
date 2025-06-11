import pickle

def dump_pkl_to_txt(pkl_path, txt_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"📂 File: {pkl_path}\n")
        f.write(f"Type: {type(data)}\n")
        f.write(f"Length: {len(data)}\n\n")

        for i, item in enumerate(data):
            f.write(f"[{i}] {item}\n")

    print(f"✅ Saved to {txt_path} (Total {len(data)} items)")

# 예시: 두 개의 pkl 파일 경로

# 두 개의 경로
pkl_path_1 = './data/avatars/wav2lip256_avatar1/coords.pkl'
pkl_path_2 = './data/avatars/wav2lip512_taeri/coords.pkl'

dump_pkl_to_txt(pkl_path_1, './coords_pkl_path_1.txt')
dump_pkl_to_txt(pkl_path_2, './coords_pkl_path_2.txt')
