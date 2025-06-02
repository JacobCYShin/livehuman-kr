'''
클래스 | 방식 | 언어 | 스트리밍 | 리샘플링 | 화자 제어
EdgeTTS | MS Edge API | 다국어 | ✔ | 필요 (sample rate mismatch) | 제한적 (voice name)
FishTTS | HTTP POST + chunk | zh | ✔ | 44100 → 16000 | 참조 기반
SovitsTTS | GPT-Sovits API | zh | ✔ | ogg 처리 필요 | 참조 기반
CosyVoiceTTS | 제로샷 API | zh | ✔ | 24000 → 16000 | 제로샷 (prompt+wav)
TencentTTS | Tencent Cloud API | zh | ✔ | 없음 | voice type
XTTS | 제로샷 서버 기반 | zh | ✔ | 24000 → 16000 | 화자 클로닝 (clone_speaker)
'''
###############################################################################
# LiveTalking 프로젝트의 TTS 처리 모듈
###############################################################################

from __future__ import annotations
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts  # Microsoft Edge TTS용 비동기 라이브러리

import os
import hmac
import hashlib
import base64
import json
import uuid
import io

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from basereal import BaseReal  # 순환 참조 방지를 위한 타입 힌트용 import

from logger import logger  # 공통 로그 객체

# 음성 재생 상태를 표현하는 Enum
class State(Enum):
    RUNNING = 0  # 재생 중
    PAUSE = 1    # 일시정지

# 모든 TTS 클래스가 상속할 기본 클래스
class BaseTTS:
    def __init__(self, opt, parent: BaseReal):
        self.opt = opt              # 설정 객체
        self.parent = parent        # 상위 시스템 (BaseReal)

        self.fps = opt.fps          # 초당 프레임 수 (20ms 기준이면 50)
        self.sample_rate = 16000    # 오디오 샘플링 레이트
        self.chunk = self.sample_rate // self.fps  # 프레임당 샘플 수 (20ms = 320샘플)
        self.input_stream = BytesIO()              # TTS 결과 저장용 임시 스트림

        self.msgqueue = Queue()     # 텍스트 메시지 입력 큐
        self.state = State.RUNNING  # 초기 상태는 실행 중

    # 입력 큐 초기화 및 일시정지
    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    # 텍스트 메시지를 큐에 추가
    def put_msg_txt(self, msg: str, eventpoint=None):
        if len(msg) > 0:
            self.msgqueue.put((msg, eventpoint))

    # TTS 처리 스레드 실행
    def render(self, quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()

    # 큐에서 메시지를 꺼내 TTS 변환 실행
    def process_tts(self, quit_event):
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state = State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)  # 하위 클래스에서 구현
        logger.info('ttsreal thread stop')

    # 실제 텍스트 → 음성 처리 함수 (Base에서는 추상적으로 정의)
    def txt_to_audio(self, msg):
        pass

class LocalTTS(BaseTTS):
    def txt_to_audio(self, msg):
        text, textevent = msg
        payload = {
            'text': text,
            'sr': 16000,
            'model': 'htr',
            'pre_post_silence_sec': 0.2,
            'intermittent_silence_sec': 0.2,
            'wav_path': 'temp/temp.wav',
            'speed': 1.0
        }

        t0 = time.time()
        res = requests.post("http://localhost:7009/api/tts", json=payload)

        print("Content-Type:", res.headers.get("Content-Type"))

        with open("debug_output.wav", "wb") as f:
            f.write(res.content)

        if res.status_code != 200:
            logger.error(f"LocalTTS 요청 실패: {res.status_code} - {res.text}")
            return

        logger.info(f"LocalTTS 요청 성공 (소요 시간: {time.time() - t0:.2f}s)")

        try:
            # 오디오 로딩 및 샘플링 레이트 확인
            audio, sample_rate = sf.read(io.BytesIO(res.content), dtype='float32')
        except Exception as e:
            logger.exception(f"오디오 로딩 실패: {e}")
            return

        logger.info(f"[LocalTTS] 오디오 shape: {audio.shape}, 샘플링 레이트: {sample_rate}")

        # === ✅ 샘플링 레이트가 다르면 리샘플링 ===
        if sample_rate != self.sample_rate:
            logger.warning(f"샘플링 레이트 불일치: {sample_rate} → {self.sample_rate}, 리샘플링 중...")
            audio = resampy.resample(audio, sr_orig=sample_rate, sr_new=self.sample_rate)
            sample_rate = self.sample_rate

        # === ✅ chunk 길이 재계산 ===
        chunk = sample_rate // self.fps

        idx = 0
        while idx + chunk <= len(audio):
            chunk_data = audio[idx:idx + chunk]
            eventpoint = None
            if idx == 0:
                eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
            elif idx + chunk >= len(audio):
                eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
            self.parent.put_audio_frame(chunk_data, eventpoint)
            logger.debug(f"[LocalTTS] 프레임 전송: idx={idx}, 이벤트={eventpoint}")
            idx += chunk


###########################################################################################
# 실제 Microsoft Edge TTS를 사용하는 클래스
class EdgeTTS(BaseTTS):
    def txt_to_audio(self, msg):
        voicename = "zh-CN-YunxiaNeural"  # 사용할 음성 이름
        text, textevent = msg             # 텍스트와 이벤트 정보 분리
        t = time.time()

        # 비동기 방식으로 TTS 실행
        asyncio.new_event_loop().run_until_complete(self.__main(voicename, text))
        logger.info(f'-------edge tts time:{time.time() - t:.4f}s')

        # 음성 생성 실패 시 로그 출력
        if self.input_stream.getbuffer().nbytes <= 0:
            logger.error('edgetts err!!!!!')
            return

        # 생성된 오디오 스트림을 numpy 배열로 변환
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx = 0

        # 320 샘플씩 나눠서 ASR 시스템으로 전달
        while streamlen >= self.chunk and self.state == State.RUNNING:
            eventpoint = None
            streamlen -= self.chunk
            if idx == 0:
                # 처음 프레임일 경우 "start" 이벤트 전달
                eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
            elif streamlen < self.chunk:
                # 마지막 프레임일 경우 "end" 이벤트 전달
                eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
            self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
            idx += self.chunk

        # 스트림 재사용을 위해 리셋
        self.input_stream.seek(0)
        self.input_stream.truncate()

    # 오디오 스트림을 numpy 배열로 변환하고 리샘플링
    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)  # float64로 로딩
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]  # 첫 번째 채널만 사용

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    # Edge TTS의 비동기 처리 본체
    async def __main(self, voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state == State.RUNNING:
                    # 생성된 오디오 데이터를 input_stream에 저장
                    self.input_stream.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass  # 단어 경계 정보는 현재 사용하지 않음
        except Exception as e:
            logger.exception('edgetts')  # 예외 발생 시 전체 트레이스 로그

class FishTTS(BaseTTS):
    def txt_to_audio(self, msg): 
        text, textevent = msg
        self.stream_tts(
            self.fish_speech(
                text,
                self.opt.REF_FILE,     # 참조 오디오 파일
                self.opt.REF_TEXT,     # 참조 텍스트
                "zh",                  # 언어 설정
                self.opt.TTS_SERVER,   # TTS 서버 주소
            ),
            msg
        )

    # FishTTS 전용 HTTP TTS 요청 함수 (streaming)
    def fish_speech(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req = {
            'text': text,
            'reference_id': reffile,
            'format': 'wav',
            'streaming': True,
            'use_memory_cache': 'on'
        }
        try:
            res = requests.post(
                f"{server_url}/v1/tts",
                json=req,
                stream=True,
                headers={"content-type": "application/json"},
            )
            end = time.perf_counter()
            logger.info(f"fish_speech Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return

            first = True
            for chunk in res.iter_content(chunk_size=17640):  # 약 20ms 분량 (44100Hz * 2ch * 20ms)
                if first:
                    end = time.perf_counter()
                    logger.info(f"fish_speech Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('fishtts')

    # 오디오 스트림을 320-sample 단위로 자르고 부모 시스템에 전달
    def stream_tts(self, audio_stream, msg):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=44100, sr_new=self.sample_rate)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        # 마지막에 'end' 이벤트 전송
        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)


class SovitsTTS(BaseTTS):
    def txt_to_audio(self, msg): 
        text, textevent = msg
        self.stream_tts(
            self.gpt_sovits(
                text,
                self.opt.REF_FILE,
                self.opt.REF_TEXT,
                "zh",
                self.opt.TTS_SERVER,
            ),
            msg
        )

    # GPT-Sovits TTS 요청 → 오디오 스트림을 생성하는 제너레이터
    def gpt_sovits(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req = {
            'text': text,
            'text_lang': language,
            'ref_audio_path': reffile,
            'prompt_text': reftext,
            'prompt_lang': language,
            'media_type': 'ogg',
            'streaming_mode': True
        }
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return

            first = True
            for chunk in res.iter_content(chunk_size=None):
                logger.info('chunk len:%d', len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('sovits')

    # 오디오 바이트 스트림을 float32로 변환하고 샘플링 레이트 맞춤
    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    # 제너레이터로 받은 오디오 데이터를 chunk 단위로 잘라서 parent에 전달
    def stream_tts(self, audio_stream, msg):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                byte_stream = BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self, msg):
        text, textevent = msg
        self.stream_tts(
            self.cosy_voice(
                text,
                self.opt.REF_FILE,
                self.opt.REF_TEXT,
                "zh",  # 언어 설정
                self.opt.TTS_SERVER,
            ),
            msg
        )

    # CosyVoice 서버에 TTS 요청을 보내고 스트리밍으로 받는 함수
    def cosy_voice(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext  # 제로샷 prompt 입력
        }
        try:
            # 참조 음성 파일 첨부
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]

            res = requests.request(
                "GET",
                f"{server_url}/inference_zero_shot",
                data=payload,
                files=files,
                stream=True
            )

            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return

            first = True
            for chunk in res.iter_content(chunk_size=9600):  # 약 20ms (24KHz 16bit mono)
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('cosyvoice')

    def stream_tts(self, audio_stream, msg):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

# 텐센트 API 관련 상수 정의
_PROTOCOL = "https://"
_HOST = "tts.cloud.tencent.com"
_PATH = "/stream"
_ACTION = "TextToStreamAudio"

class TencentTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        # Tencent API 인증 정보 (환경 변수에서 불러옴)
        self.appid = os.getenv("TENCENT_APPID")
        self.secret_key = os.getenv("TENCENT_SECRET_KEY")
        self.secret_id = os.getenv("TENCENT_SECRET_ID")
        self.voice_type = int(opt.REF_FILE)  # 음성 모델 ID
        self.codec = "pcm"
        self.sample_rate = 16000
        self.volume = 0
        self.speed = 0

    # Tencent Cloud용 서명 생성 함수 (SHA1 + Base64)
    def __gen_signature(self, params):
        sort_dict = sorted(params.keys())
        sign_str = "POST" + _HOST + _PATH + "?"
        for key in sort_dict:
            sign_str += f"{key}={params[key]}&"
        sign_str = sign_str[:-1]  # 마지막 & 제거
        hmacstr = hmac.new(self.secret_key.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha1).digest()
        return base64.b64encode(hmacstr).decode('utf-8')

    # TTS 요청에 필요한 파라미터 구성
    def __gen_params(self, session_id, text):
        params = {
            'Action': _ACTION,
            'AppId': int(self.appid),
            'SecretId': self.secret_id,
            'ModelType': 1,
            'VoiceType': self.voice_type,
            'Codec': self.codec,
            'SampleRate': self.sample_rate,
            'Speed': self.speed,
            'Volume': self.volume,
            'SessionId': session_id,
            'Text': text,
            'Timestamp': int(time.time()),
            'Expired': int(time.time()) + 86400  # 유효기간 24시간
        }
        return params

    def txt_to_audio(self, msg):
        text, textevent = msg
        self.stream_tts(
            self.tencent_voice(
                text,
                self.opt.REF_FILE,
                self.opt.REF_TEXT,
                "zh",
                self.opt.TTS_SERVER,
            ),
            msg
        )

    # Tencent TTS 서버에 요청 후 PCM 오디오를 스트리밍으로 받음
    def tencent_voice(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        session_id = str(uuid.uuid1())
        params = self.__gen_params(session_id, text)
        signature = self.__gen_signature(params)
        headers = {
            "Content-Type": "application/json",
            "Authorization": str(signature)
        }
        url = _PROTOCOL + _HOST + _PATH

        try:
            res = requests.post(url, headers=headers, data=json.dumps(params), stream=True)

            end = time.perf_counter()
            logger.info(f"tencent Time to make POST: {end-start}s")

            first = True
            for chunk in res.iter_content(chunk_size=6400):  # 약 20ms
                if first:
                    try:
                        rsp = json.loads(chunk)
                        logger.error("tencent tts:%s", rsp["Response"]["Error"]["Message"])
                        return
                    except:
                        end = time.perf_counter()
                        logger.info(f"tencent Time to first chunk: {end-start}s")
                        first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('tencent')

    def stream_tts(self, audio_stream, msg):
        text, textevent = msg
        first = True
        last_stream = np.array([], dtype=np.float32)
        for chunk in audio_stream:
            if chunk and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream, stream))
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:]  # 남은 데이터는 다음 chunk에 이어붙임
        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        # 참조 오디오를 사용해 speaker embedding을 생성하여 저장
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self, msg):
        text, textevent = msg
        self.stream_tts(
            self.xtts(
                text,
                self.speaker,  # clone_speaker로부터 받은 화자 정보 포함 JSON
                "zh-cn",       # 언어 코드
                self.opt.TTS_SERVER,
                "20"           # stream_chunk_size (응답 빠르게 받을 수 있음)
            ),
            msg
        )

    # 참조 음성을 기반으로 speaker embedding 생성
    def get_speaker(self, ref_audio, server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()  # 화자 프로필 반환

    # XTTS 서버로 요청을 보내고 오디오 스트림을 생성하는 제너레이터 함수
    def xtts(self, text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
            for chunk in res.iter_content(chunk_size=9600):  # 24000Hz, 20ms 단위
                if first:
                    end = time.perf_counter()
                    logger.info(f"xtts Time to first chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)

    # 오디오 스트림을 받아 chunk 단위로 잘라 WebRTC(TTS 시스템)에 전달
    def stream_tts(self, audio_stream, msg):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                # int16 → float32로 변환 후 리샘플링 (24kHz → 16kHz)
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        # 마지막 빈 프레임으로 "end" 이벤트 전달
        eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

