'''
                        ┌────────────────────┐
                        │  Client (Browser)  │
                        │  - WebRTC Dashboard│
                        │  - API Controller  │
                        └────────┬───────────┘
                                 │ Offer SDP
                                 ▼
                    ┌────────────────────────────┐
                    │        aiohttp Server      │
                    │     (WebRTC + REST API)    │
                    └────────┬──────────┬────────┘
                             │          │
         ┌───────────────────┘          └─────────────────────┐
         ▼                                            ▼
┌──────────────────────┐                    ┌────────────────────────┐
│  build_nerfreal()    │                    │  REST API Endpoints     │
│  └── Load Model      │                    │  - /human               │
│  └── Load Avatar     │                    │  - /humanaudio          │
│  └── Warm-up         │                    │  - /set_audiotype       │
└─────────┬────────────┘                    │  - /record              │
          │                                 │  - /is_speaking         │
          ▼                                 └──────────┬──────────────┘
 ┌────────────────────────┐                            │
 │  LipReal / MuseReal /  │                            ▼
 │  NeRFReal (BaseReal)   │                ┌─────────────────────────────┐
 │  └── ASR (LipASR)      │                │    TTS (EdgeTTS, XTTS 등)    │
 │  └── TTS (BaseTTS)     │                └────────────┬────────────────┘
 │  └── inference()       │                             │
 │  └── render()          │ <────── call ───────────────┘
 └──────┬─────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│         WebRTC PeerConnection (aiortc)       │
│  ┌────────────────────────────────────────┐  │
│  │ audio_track ← HumanPlayer.audio       │  │
│  │ video_track ← HumanPlayer.video       │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘

구성요소 | 역할
Client | WebRTC 대시보드 or API 요청자
aiohttp | WebRTC offer 처리 및 REST API
build_nerfreal() | 모델/아바타 로딩 및 디지털 휴먼 인스턴스 생성
BaseReal 계열 | LipReal, MuseReal 등 디지털 휴먼 로직
ASR, TTS | 오디오 분석 및 합성 (입력 텍스트 → 음성)
inference() | Wav2Lip 등 입 모양 생성 (음성 + 얼굴)
render() | 오디오/비디오 WebRTC 트랙으로 푸시
HumanPlayer | aiortc용 오디오/비디오 track 생성기
WebRTC PC | 브라우저와 미디어 트랙 송수신

'''
# Windows 환경에서 UTF-8 인코딩을 강제로 설정 (터미널 한글 출력 깨짐 방지용)
# chcp 65001

# 실행 예시: 특정 모델 및 아바타로 app.py 실행
# python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
# python app.py --transport webrtc --model wav2lip --avatar_id wav2lip512_taeri
'''
conda activate nerfstream
chcp 65001 
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip_taeri
'''
###############################################################################
#  LiveTalking 프로젝트 라이선스 및 저작권 정보
###############################################################################

# 서버 백엔드 기본 라이브러리 임포트
from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_sockets import Sockets  # WebSocket 지원용 (현재는 사용 안 함)
import base64
import json
import re
import numpy as np
from threading import Thread, Event
import torch.multiprocessing as mp  # PyTorch 멀티프로세싱 사용

# WebRTC 및 실시간 스트리밍 관련 라이브러리
from aiohttp import web
import aiohttp
import aiohttp_cors  # CORS 설정을 위한 aiohttp 확장
from aiortc import RTCPeerConnection, RTCSessionDescription  # WebRTC 연결 구성 요소
from aiortc.rtcrtpsender import RTCRtpSender

# 내부 모듈 - 디지털 휴먼 로직
from webrtc import HumanPlayer  # 오디오/비디오 스트리밍 핸들러
from basereal import BaseReal    # 디지털휴먼 기반 클래스
from llm import llm_response     # LLM 기반 응답 생성 함수

# 기타 유틸 라이브러리
import argparse  # 커맨드라인 옵션 파싱
import random
import shutil
import asyncio
import torch
from typing import Dict
from logger import logger  # 커스텀 로깅 유틸

# Flask 앱 생성
app = Flask(__name__)

# 디지털 휴먼 인스턴스를 sessionid 별로 저장하는 딕셔너리
nerfreals: Dict[int, BaseReal] = {}

# 실행 시 옵션 값과 모델, 아바타 객체 저장용 전역 변수
opt = None
model = None
avatar = None

##### WebRTC 관련 전역 세션 목록 #####
pcs = set()  # 현재 연결된 모든 WebRTC PeerConnection을 저장

# N자리의 무작위 숫자를 생성하는 함수 (세션 ID 용)
def randN(N) -> int:
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

# 세션별 디지털 휴먼 객체 생성 함수
def build_nerfreal(sessionid: int) -> BaseReal:
    opt.sessionid = sessionid
    # 선택된 모델 종류에 따라 서로 다른 클래스 로드
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt, model, avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt, model, avatar)
    elif opt.model == 'ernerf':
        from nerfreal import NeRFReal
        nerfreal = NeRFReal(opt, model, avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt, model, avatar)
    return nerfreal

# WebRTC offer 요청을 처리하는 비동기 HTTP 핸들러
# WebRTC에서 offer 요청은 "나 연결할 준비 됐는데, 너는 어때?" 라는 식의 초기 연결 제안 메시지입니다. 이건 WebRTC의 시그널링(Signaling) 과정의 핵심
'''
[1] 유저가 offer 요청 (브라우저)
    │
    ▼
[2] 서버에서 offer 처리 (offer 함수)
    │
    ├─[a] 디지털 휴먼 객체 생성 (build_nerfreal)
    │       ↓
    │     HumanPlayer(nerfreals[sessionid])
    │       ├─ audio (가상 마이크)
    │       └─ video (가상 카메라)
    │
    ├─[b] WebRTC 연결 생성 (RTCPeerConnection)
    │       ├─ pc.addTrack(audio)
    │       └─ pc.addTrack(video)
    │
    ├─[c] 코덱 우선순위 설정 (setCodecPreferences)
    │
    ├─[d] answer 생성 및 클라이언트에 전송
    │
    ▼
[3] 클라이언트에서 answer 수신 → WebRTC 연결 완료
    │
    ▼
[4] 디지털 휴먼 오디오/비디오가 실시간 전송됨
    │
    ▼
[5] 연결 종료/실패 감지 시:
    └─ connectionStatechange 콜백 → 자원 정리

'''
async def offer(request):
    # 클라이언트로부터 SDP offer 수신
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # 최대 세션 수 초과 시 연결 거부
    if len(nerfreals) >= opt.max_session:
        logger.info('reach max session')
        return -1

    # 새로운 세션 ID 생성
    sessionid = randN(6)
    logger.info('sessionid=%d', sessionid)

    # 디지털 휴먼 객체 초기화 (백그라운드 스레드로 실행)
    nerfreals[sessionid] = None
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal

    # WebRTC 연결 객체 생성
    pc = RTCPeerConnection()
    pcs.add(pc)

    # 연결 상태 변경 콜백 등록
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        # 연결 종료 또는 실패 시 cleanup
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            del nerfreals[sessionid]

    # 디지털 휴먼으로부터 오디오/비디오 트랙 생성
    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    # 사용 가능한 비디오 코덱 설정
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    # 클라이언트 offer 설정 적용 및 응답 answer 생성
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    # ICE란? : 연결 가능한 IP/포트 조합, 연결 경로 후보들

    # 클라이언트에 answer + sessionid 응답 반환
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "sessionid": sessionid
        }),
    )

# 사용자의 텍스트 요청을 처리하는 엔드포인트 (echo 또는 chat)
async def human(request):
    # raw HTTP body (bytes) → JSON 문자열 → Python dict
    params = await request.json()  # 요청 바디에서 JSON 파싱

    sessionid = params.get('sessionid', 0)

    # 사용자가 중간에 끊기를 요청한 경우: 현재 음성 출력 취소
    if params.get('interrupt'):
        nerfreals[sessionid].flush_talk()

    # echo 타입: 입력 텍스트를 그대로 출력
    if params['type'] == 'echo':
        nerfreals[sessionid].put_msg_txt(params['text'])

    # chat 타입: LLM을 통해 응답 생성
    elif params['type'] == 'chat':
        res = await asyncio.get_event_loop().run_in_executor(
            None, llm_response, params['text'], nerfreals[sessionid]
        )
        # (선택) 생성된 응답을 speak queue에 넣으려면 아래 라인 활성화
        # nerfreals[sessionid].put_msg_txt(res)
    ''' await asyncio.get_event_loop().run_in_executor 추가가 설명
    [메인 asyncio 루프]     ─────────────┐
      ↓                           ↓
    await run_in_executor(...)   →  [백그라운드 스레드] → some_function() 실행
        ↓                           ↑
    다른 코루틴도 실행 중        작업 완료되면 결과 반환
    '''

    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"})
    )


# 사용자가 업로드한 오디오 파일을 처리하는 엔드포인트
async def humanaudio(request):
    try:
        form = await request.post()  # multipart/form-data 파싱
        sessionid = int(form.get('sessionid', 0))

        fileobj = form["file"]  # 파일 객체 획득
        filename = fileobj.filename
        filebytes = fileobj.file.read()  # 바이트로 읽기

        # 디지털휴먼 객체에 오디오 바이트 전달
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"})
        )
    except Exception as e:
        # 에러 발생 시 에러 메시지 포함 응답
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "err", "data": str(e)})
        )


# 현재 오디오 타입 (예: 실시간 / 참조 기반 등) 설정
async def set_audiotype(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)

    # 오디오 상태 및 초기화 여부 전달
    nerfreals[sessionid].set_curr_state(params['audiotype'], params['reinit'])

    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"})
    )


# 녹음 시작/종료 요청 처리
async def record(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)

    if params['type'] == 'start_record':
        nerfreals[sessionid].start_recording()
    elif params['type'] == 'end_record':
        nerfreals[sessionid].stop_recording()

    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"})
    )


# 현재 디지털휴먼이 말하고 있는 상태인지 여부 반환
async def is_speaking(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "code": 0,
            "data": nerfreals[sessionid].is_speaking()
        })
    )


# 서버 종료 시 모든 WebRTC 연결 종료 처리
async def on_shutdown(app):
    # 모든 peer connection 종료
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


# 외부 서버에 POST 요청을 비동기로 보내는 유틸 함수
async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')


# 외부 push_url로 WebRTC offer → answer 처리하는 단독 실행 흐름 (비표준용도)
async def run(push_url, sessionid):
    # 디지털 휴먼 생성
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # 디지털휴먼으로부터 오디오/비디오 트랙 생성
    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    # offer 생성 → remote answer 수신 → 연결 완료
    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))


# (참고) 시스템 환경변수 설정 예시 - Intel MKL 강제 사용 등
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'

                                           
if __name__ == '__main__':
    # PyTorch 멀티프로세싱 초기화 방식 설정 (Windows 호환 위해 'spawn' 사용)
    mp.set_start_method('spawn')

    # 명령줄 인자 설정
    parser = argparse.ArgumentParser()

    # 포즈, 얼굴 근육(눈 깜빡임), 상체 이미지 경로 설정
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")

    # -O는 여러 최적화 옵션을 한꺼번에 적용 (fp16 + cuda_ray + exp_eye)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")

    # 학습에 사용할 데이터 범위
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)

    ### 학습 관련 옵션
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')  # 사전학습 체크포인트

    # 레이 샘플링 관련 (NeRF 관련 옵션)
    parser.add_argument('--num_rays', type=int, default=4096 * 16)
    parser.add_argument('--cuda_ray', action='store_true')
    parser.add_argument('--max_steps', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=16)
    parser.add_argument('--upsample_steps', type=int, default=0)
    parser.add_argument('--update_extra_interval', type=int, default=16)
    parser.add_argument('--max_ray_batch', type=int, default=4096)

    ### 손실 함수 설정
    parser.add_argument('--warmup_step', type=int, default=10000)
    parser.add_argument('--amb_aud_loss', type=int, default=1)
    parser.add_argument('--amb_eye_loss', type=int, default=1)
    parser.add_argument('--unc_loss', type=int, default=1)
    parser.add_argument('--lambda_amb', type=float, default=1e-4)

    ### 네트워크 백본 관련
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bg_img', type=str, default='white')
    parser.add_argument('--fbg', action='store_true')
    parser.add_argument('--exp_eye', action='store_true')
    parser.add_argument('--fix_eye', type=float, default=-1)
    parser.add_argument('--smooth_eye', action='store_true')
    parser.add_argument('--torso_shrink', type=float, default=0.8)

    ### 데이터셋 로딩 설정
    parser.add_argument('--color_space', type=str, default='srgb')
    parser.add_argument('--preload', type=int, default=0)
    parser.add_argument('--bound', type=float, default=1)
    parser.add_argument('--scale', type=float, default=4)
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0])
    parser.add_argument('--dt_gamma', type=float, default=1/256)
    parser.add_argument('--min_near', type=float, default=0.05)
    parser.add_argument('--density_thresh', type=float, default=10)
    parser.add_argument('--density_thresh_torso', type=float, default=0.01)
    parser.add_argument('--patch_size', type=int, default=1)

    parser.add_argument('--init_lips', action='store_true')
    parser.add_argument('--finetune_lips', action='store_true')
    parser.add_argument('--smooth_lips', action='store_true')

    parser.add_argument('--torso', action='store_true')
    parser.add_argument('--head_ckpt', type=str, default='')

    ### GUI 관련 옵션
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--W', type=int, default=450)
    parser.add_argument('--H', type=int, default=450)
    parser.add_argument('--radius', type=float, default=3.35)
    parser.add_argument('--fovy', type=float, default=21.24)
    parser.add_argument('--max_spp', type=int, default=1)

    ### 오디오 / attention / 개별 코드 관련
    parser.add_argument('--att', type=int, default=2)
    parser.add_argument('--aud', type=str, default='')
    parser.add_argument('--emb', action='store_true')
    parser.add_argument('--ind_dim', type=int, default=4)
    parser.add_argument('--ind_num', type=int, default=10000)
    parser.add_argument('--ind_dim_torso', type=int, default=8)
    parser.add_argument('--amb_dim', type=int, default=2)
    parser.add_argument('--part', action='store_true')
    parser.add_argument('--part2', action='store_true')

    parser.add_argument('--train_camera', action='store_true')
    parser.add_argument('--smooth_path', action='store_true')
    parser.add_argument('--smooth_path_window', type=int, default=7)

    ### 실시간 음성 인식 (ASR) 옵션
    parser.add_argument('--asr', action='store_true')
    parser.add_argument('--asr_wav', type=str, default='')
    parser.add_argument('--asr_play', action='store_true')
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto')
    parser.add_argument('--asr_save_feats', action='store_true')

    # 오디오 fps 및 슬라이딩 윈도우 설정
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    ### 전신 아바타 설정
    parser.add_argument('--fullbody', action='store_true')
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)

    # musetalk 옵션
    parser.add_argument('--avatar_id', type=str, default='avator_1')
    parser.add_argument('--bbox_shift', type=int, default=5) # only Musetalk
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--customvideo_config', type=str, default='')

    # 음성 합성 (TTS) 관련
    parser.add_argument('--tts', type=str, default='local')
    parser.add_argument('--REF_FILE', type=str, default='reference.wav')
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:7009')

    # 모델 및 전송 방식 선택
    parser.add_argument('--model', type=str, default='ernerf')
    parser.add_argument('--transport', type=str, default='rtcpush')
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010)

    # 인자 파싱
    opt = parser.parse_args()

    # custom video 설정 json 불러오기
    opt.customopt = []
    if opt.customvideo_config != '':
        with open(opt.customvideo_config, 'r') as file:
            opt.customopt = json.load(file)

    # 모델에 따라 동적으로 모듈 import 및 로딩
    if opt.model == 'ernerf':
        from nerfreal import NeRFReal, load_model, load_avatar
        model = load_model(opt)
        avatar = load_avatar(opt)

    elif opt.model == 'musetalk':
        from musereal import MuseReal, load_model, load_avatar, warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model)

    elif opt.model == 'wav2lip':
        from lipreal import LipReal, load_model, load_avatar, warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, model, 256)
        # warm_up(opt.batch_size, model, 512)

    elif opt.model == 'ultralight':
        from lightreal import LightReal, load_model, load_avatar, warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size, avatar, 160)

    # RTMP 모드일 경우 별도 스레드로 렌더링 시작
    if opt.transport == 'rtmp':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()

    # 웹 서버 초기화
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)

    # API 라우팅 등록
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_static('/', path='web')

    # CORS 허용 설정
    cors = aiohttp_cors.setup(appasync, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(appasync.router.routes()):
        cors.add(route)

    # transport 모드에 따라 기본 페이지 선택
    pagename = {
        'rtmp': 'echoapi.html',
        'rtcpush': 'rtcpushapi.html'
    }.get(opt.transport, 'webrtcapi.html')

    # 웹 서버 시작 로그 출력 (서버 접속 URL 안내)
    logger.info(f"📡 HTTP 서버 시작: http://<serverip>:{opt.listenport}/{pagename}")
    logger.info(f"💡 WebRTC 사용 시 대시보드 접속 권장: http://<serverip>:{opt.listenport}/dashboard.html")

    # 웹서버 실행을 위한 비동기 함수 정의
    def run_server(runner):
        # 별도의 이벤트 루프 생성 및 설정
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 앱 실행 준비 (라우팅, 미들웨어, etc)
        loop.run_until_complete(runner.setup())

        # 실제 TCP 서버 바인딩 및 시작 (listenport로 수신 대기)
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())

        # transport 방식이 'rtcpush'일 경우: 지정된 push_url로 세션별 비디오 푸시
        if opt.transport == 'rtcpush':
            for k in range(opt.max_session):
                # 세션 번호에 따라 URL 구분 (stream1, stream2 등)
                push_url = opt.push_url if k == 0 else opt.push_url + str(k)
                # 각 세션별로 run(push_url, sessionid) 호출 → WebRTC 연결 시작
                loop.run_until_complete(run(push_url, k))

        # 서버 무한 루프 시작 (신호가 들어올 때까지 계속 대기)
        loop.run_forever()

    # aiohttp 웹 애플리케이션 실행
    run_server(web.AppRunner(appasync))
