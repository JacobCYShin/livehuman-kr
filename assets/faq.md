⚙️ 문제 해결 가이드
1. PyTorch3D 설치 실패
소스코드를 직접 빌드하여 설치합니다:


git clone https://github.com/facebookresearch/pytorch3d.git
python setup.py install
2. WebSocket 연결 오류
python/site-packages/flask_sockets.py 파일을 아래와 같이 수정합니다:

python

# 수정 전
self.url_map.add(Rule(rule, endpoint=f))

# 수정 후
self.url_map.add(Rule(rule, endpoint=f, websocket=True))
3. Protobuf 버전 오류
protobuf 버전이 너무 높을 경우, 아래와 같이 버전을 다운그레이드합니다:


pip uninstall protobuf
pip install protobuf==3.20.1
4. 디지털 휴먼 눈 깜빡임(AU45) 미작동
OpenFace의 FeatureExtraction을 실행하여 AU45 값을 얻습니다.


# OpenFace 실행 후 출력된 CSV 파일을 다음과 같이 이동
mv output.csv data/<ID>/au.csv
반드시 au.csv 파일을 data/<ID>/ 경로에 위치시켜야 합니다.

5. 디지털 휴먼에 배경 이미지 추가

python app.py --bg_img bc.jpg
6. 사용자 모델 적용 시 차원 불일치 오류
ASR 모델로 wav2vec2 사용 시 다음과 같이 학습해야 합니다:


python main.py data/ --workspace workspace/ -O --iters 100000 \
--asr_model cpierse/wav2vec2-large-xlsr-53-esperanto
7. RTMP 스트리밍 시 FFmpeg 버전 문제
FFmpeg는 libx264가 활성화되어 있어야 합니다.

일부 사용자들은 v4.2.2 버전에서 정상 작동을 확인했습니다.


ffmpeg -version | grep libx264
# 출력에 --enable-libx264 포함 여부 확인
8. 사용자 학습 모델 구조
text

.
├── data
│   ├── data_kf.json        # transforms_train.json에 대응
│   ├── au.csv              # 눈 깜빡임 (AU45) 정보
│   └── pretrained
│       └── ngp_kf.pth      # 학습된 모델 (예: ngp_ep00xx.pth)
📎 참고 자료
관련 GitHub 이슈 코멘트