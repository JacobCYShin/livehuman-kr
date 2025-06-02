'''
                         ┌─────────────┐
       Audio In         │  LipASR.run_step() ─────┐
        (stream)        └─────────────┘           ▼
                                               Mel + PCM
                                                  ▼
                      ┌────────────┐       ┌─────────────┐
                      │ inference()│ ◀──── │ face images │
                      └────────────┘       └─────────────┘
                             ▼
                      합성된 입 프레임
                             ▼
          ┌────────────┐  좌표로 덮기   ┌────────────┐
          │ process_frames │──────────▶│ WebRTC 송출 │
          └────────────┘               └────────────┘

'''

###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# 기본 라이브러리
import math
import torch
import numpy as np

# 기타 유틸
import os
import time
import cv2
import glob
import pickle
import copy

# 큐 및 멀티스레딩/프로세싱 관련
import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

# 음성 인식 + Wav2Lip + 통합 처리 시스템
from lipasr import LipASR
import asyncio
from av import AudioFrame, VideoFrame
from wav2lip.models import Wav2Lip
from basereal import BaseReal

from tqdm import tqdm
from logger import logger  # 로그 출력용

# 사용 가능한 디바이스 설정 (CUDA > MPS > CPU)
device = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
print('Using {} for inference.'.format(device))

# Wav2Lip 모델 checkpoint 로딩
def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	return checkpoint

# Wav2Lip 모델 초기화 및 가중치 적용
def load_model(path):
	model = Wav2Lip()
	logger.info("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v  # multi-GPU 모델 키 수정
	model.load_state_dict(new_s)
	model = model.to(device)
	return model.eval()

# 아바타 데이터 (프레임/좌표) 로드
def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"

    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)

    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle, face_list_cycle, coord_list_cycle

# 모델 warm-up을 위한 더미 입력 실행
@torch.no_grad()
def warm_up(batch_size, model, modelres):
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

# 이미지 목록을 cv2로 읽고 리스트로 반환
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

# 프레임 반복 시 좌우 왕복 방식으로 index 순환
def __mirror_index(size, index):
    turn = index // size
    res = index % size
    return res if turn % 2 == 0 else size - res - 1

# 메인 입모양 합성 처리 루프
def inference(quit_event, batch_size, face_list_cycle, audio_feat_queue, audio_out_queue, res_frame_queue, model):
    length = len(face_list_cycle)
    index = 0
    count = 0
    counttime = 0
    logger.info('start inference')

    while not quit_event.is_set():
        starttime = time.perf_counter()
        mel_batch = []

        # 오디오 특징(Mel spectrogram) 받아오기
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue

        is_all_silence = True
        audio_frames = []

        # 오디오 프레임 받아오기 (2개당 1프레임)
        for _ in range(batch_size * 2):
            frame, type, eventpoint = audio_out_queue.get()
            audio_frames.append((frame, type, eventpoint))
            if type == 0:  # 0이면 실제 발화
                is_all_silence = False

        if is_all_silence:
            # 전부 무음이면 빈 프레임 처리
            for i in range(batch_size):
                res_frame_queue.put((None, __mirror_index(length, index), audio_frames[i*2:i*2+2]))
                index += 1
        else:
            # 얼굴 이미지와 멜스펙트로그램을 batch로 구성
            t = time.perf_counter()
            img_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length, index+i)
                face = face_list_cycle[idx]
                img_batch.append(face)
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            # 마스크 처리 (하단 반은 0 처리)
            img_masked = img_batch.copy()
            img_masked[:, face.shape[0]//2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0

            # mel: [B, 1, 80, 16] / image: [B, 6, H, W]
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0  # [B, H, W, C]

            counttime += (time.perf_counter() - t)
            count += batch_size
            if count >= 100:
                logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                count = 0
                counttime = 0

            # 결과 프레임 큐에 삽입
            for i, res_frame in enumerate(pred):
                res_frame_queue.put((res_frame, __mirror_index(length, index), audio_frames[i*2:i*2+2]))
                index += 1

    logger.info('lipreal inference processor stop')

class LipReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)  # BaseReal 초기화 (tts, asr 등 기본 요소 포함)
        self.W = opt.W  # 영상 가로 크기
        self.H = opt.H  # 영상 세로 크기

        self.fps = opt.fps  # 프레임 속도 (ex: 50 = 20ms 단위)

        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size * 2)  # 합성 결과 프레임 저장 큐

        self.model = model  # 미리 로드된 Wav2Lip 모델
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar  # 아바타 이미지, 얼굴, 좌표

        self.asr = LipASR(opt, self)  # 입모양 동기화를 위한 ASR 인식기 생성
        self.asr.warm_up()  # ASR warm-up

        self.render_event = mp.Event()  # (미사용) 렌더링 상태 sync용 이벤트 객체

    def __del__(self):
        logger.info(f'lipreal({self.sessionid}) delete')  # 인스턴스 종료 시 로그 출력

    # 비디오/오디오 프레임을 가져와 WebRTC 스트림에 송출하는 함수
    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # 🔇 무음 상태이면: full 이미지만 보여주고, 음성은 None 처리
            if audio_frames[0][1] != 0 and audio_frames[1][1] != 0:
                self.speaking = False
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None:
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]), self.custom_index[audiotype])
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    combine_frame = self.frame_list_cycle[idx]
            else:
                # 🗣 발화 상태이면: 합성된 입 프레임을 얼굴 위에 덮어씀
                self.speaking = True
                bbox = self.coord_list_cycle[idx]
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                y1, y2, x1, x2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                except:
                    continue
                combine_frame[y1:y2, x1:x2] = res_frame  # 입만 합성

            # 영상 프레임을 WebRTC 전송
            image = combine_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame, None)), loop)
            self.record_video_data(image)  # 저장 (옵션)

            # 오디오도 WebRTC 전송
            for audio_frame in audio_frames:
                frame, type, eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame, eventpoint)), loop)
                self.record_audio_data(frame)  # 저장 (옵션)

        logger.info('lipreal process_frames thread stop')

    # 전체 렌더링 실행 함수: TTS + inference + frame 처리 thread 실행
    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        self.tts.render(quit_event)  # TTS 쓰레드 시작
        self.init_customindex()  # 커스텀 영상 index 초기화

        # 비디오/오디오 프레임 처리 쓰레드 시작
        process_thread = Thread(target=self.process_frames, args=(quit_event, loop, audio_track, video_track))
        process_thread.start()

        # Wav2Lip 추론(inference) 처리 쓰레드 시작
        Thread(target=inference, args=(
            quit_event,
            self.batch_size,
            self.face_list_cycle,
            self.asr.feat_queue,
            self.asr.output_queue,
            self.res_frame_queue,
            self.model,
        )).start()

        # 매 프레임마다 run_step으로 ASR 업데이트 + 큐 상태 보고 sleep
        count = 0
        totaltime = 0
        _starttime = time.perf_counter()
        while not quit_event.is_set():
            t = time.perf_counter()
            self.asr.run_step()  # 오디오 입력 처리

            # video_track 큐가 너무 많으면 sleep으로 제어 (버퍼링 방지)
            if video_track._queue.qsize() >= 5:
                logger.debug('sleep qsize=%d', video_track._queue.qsize())
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)
                

        logger.info('lipreal thread stop')

