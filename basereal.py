###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Apache License 2.0 하에 소스코드 공개
###############################################################################

import math
import torch
import numpy as np

import subprocess
import os
import time
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf  # 오디오 파일 읽기 및 저장

import av  # FFmpeg을 Python에서 다루기 위한 라이브러리
from fractions import Fraction

# 다양한 TTS 엔진들 불러오기
from ttsreal import EdgeTTS, SovitsTTS, XTTS, CosyVoiceTTS, FishTTS, TencentTTS, LocalTTS
from logger import logger

from tqdm import tqdm  # 프로그레스 바 출력용

# 이미지 리스트를 받아서 numpy 배열 형태로 로딩하는 함수
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):  # 이미지 로딩 과정을 시각적으로 표시
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

# LiveTalking의 실시간 TTS/ASR 시스템의 기본 클래스
class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000  # 오디오 샘플링 레이트
        self.chunk = self.sample_rate // opt.fps  # 프레임 단위 오디오 청크 크기 (예: 20ms = 320샘플)
        self.sessionid = self.opt.sessionid  # 현재 세션 식별자

        # 선택된 TTS 엔진에 따라 객체 초기화
        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt, self)
        elif opt.tts == "gpt-sovits":
            self.tts = SovitsTTS(opt, self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt, self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt, self)
        elif opt.tts == "fishtts":
            self.tts = FishTTS(opt, self)
        elif opt.tts == "tencent":
            self.tts = TencentTTS(opt, self)
        elif opt.tts == "local":
            self.tts = LocalTTS(opt, self)

        self.speaking = False  # 현재 TTS가 말하고 있는지 여부

        self.recording = False  # 영상/음성 녹음 여부
        self._record_video_pipe = None  # 영상 파이프 핸들러
        self._record_audio_pipe = None  # 오디오 파이프 핸들러
        self.width = self.height = 0  # 영상 해상도 정보

        self.curr_state = 0  # 사용자 정의 음성 상태 (0: 없음)
        self.custom_img_cycle = {}    # 사용자 정의 이미지 리스트
        self.custom_audio_cycle = {}  # 사용자 정의 오디오 배열
        self.custom_audio_index = {}  # 사용자 정의 오디오 위치 인덱스
        self.custom_index = {}        # 사용자 정의 이미지 위치 인덱스
        self.custom_opt = {}          # 사용자 정의 설정 저장

        self.__loadcustom()  # 사용자 정의 리소스 로드

    # 텍스트 메시지를 TTS 모듈에 전달
    def put_msg_txt(self, msg, eventpoint=None):
        self.tts.put_msg_txt(msg, eventpoint)

    # 오디오 프레임 (16kHz, 20ms 단위 PCM) 을 ASR 모듈로 전달
    def put_audio_frame(self, audio_chunk, eventpoint=None):
        self.asr.put_audio_frame(audio_chunk, eventpoint)

    # 오디오 파일 전체 (bytes)를 받아 청크 단위로 나누어 ASR에 입력
    def put_audio_file(self, filebyte): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk:
            self.put_audio_frame(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk

    # BytesIO 기반 스트림을 numpy 배열로 변환하고 샘플링 레이트를 일치시킴
    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)  # 오디오 로딩 (float64)
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)  # float32로 변환

        if stream.ndim > 1:  # 스테레오인 경우
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]  # 첫 번째 채널만 사용

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    # TTS 및 ASR의 입력 큐 초기화
    def flush_talk(self):
        self.tts.flush_talk()
        self.asr.flush_talk()

    # 현재 TTS가 말하고 있는지 확인
    def is_speaking(self) -> bool:
        return self.speaking

    # 사용자 정의 이미지/오디오 로딩 (예: 특정 이벤트 발생 시 사용할 음성/영상)
    def __loadcustom(self):
        for item in self.opt.customopt:
            logger.info(item)
            # 이미지 파일 리스트 로드 (숫자 순 정렬)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)

            # 오디오 파일 로드
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    # 사용자 정의 음성/이미지 인덱스를 초기화
    def init_customindex(self):
        self.curr_state = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    # 외부 이벤트 알림 로그
    def notify(self, eventpoint):
        logger.info("notify:%s", eventpoint)


    def start_recording(self):
        """비디오 녹화 시작"""
        '''
        실시간 인코딩 가능 → 녹화 중단 시점까지 진행된 데이터는 이미 저장됨

        메모리 효율적 → 대용량 오디오/영상 전체를 RAM에 들고 있을 필요 없음

        스레드/비동기 방식으로 병렬 처리 가능 → 실시간 TTS나 렌더링과 함께 진행 가능
        '''
        if self.recording:
            return

        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24', #像素格式
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    #'-f' , 'flv',                  
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    #'-acodec','pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    #'-f' , 'wav',                  
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        # self.recordq_video.queue.clear()
        # self.recordq_audio.queue.clear()
        # self.container = av.open(path, mode="w")
    
        # process_thread = Thread(target=self.record_frame, args=())
        # process_thread.start()
    
    def record_video_data(self,image):
        if self.width == 0:
            print("image.shape:",image.shape)
            self.height,self.width,_ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self,frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
    
    # def record_frame(self): 
    #     videostream = self.container.add_stream("libx264", rate=25)
    #     videostream.codec_context.time_base = Fraction(1, 25)
    #     audiostream = self.container.add_stream("aac")
    #     audiostream.codec_context.time_base = Fraction(1, 16000)
    #     init = True
    #     framenum = 0       
    #     while self.recording:
    #         try:
    #             videoframe = self.recordq_video.get(block=True, timeout=1)
    #             videoframe.pts = framenum #int(round(framenum*0.04 / videostream.codec_context.time_base))
    #             videoframe.dts = videoframe.pts
    #             if init:
    #                 videostream.width = videoframe.width
    #                 videostream.height = videoframe.height
    #                 init = False
    #             for packet in videostream.encode(videoframe):
    #                 self.container.mux(packet)
    #             for k in range(2):
    #                 audioframe = self.recordq_audio.get(block=True, timeout=1)
    #                 audioframe.pts = int(round((framenum*2+k)*0.02 / audiostream.codec_context.time_base))
    #                 audioframe.dts = audioframe.pts
    #                 for packet in audiostream.encode(audioframe):
    #                     self.container.mux(packet)
    #             framenum += 1
    #         except queue.Empty:
    #             print('record queue empty,')
    #             continue
    #         except Exception as e:
    #             print(e)
    #             #break
    #     for packet in videostream.encode(None):
    #         self.container.mux(packet)
    #     for packet in audiostream.encode(None):
    #         self.container.mux(packet)
    #     self.container.close()
    #     self.recordq_video.queue.clear()
    #     self.recordq_audio.queue.clear()
    #     print('record thread stop')
		
    def stop_recording(self):
        """녹화 중지지"""
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()  #wait() 
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        cmd_combine_audio = f"ffmpeg -y -i temp{self.opt.sessionid}.aac -i temp{self.opt.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        os.system(cmd_combine_audio) 
        #os.remove(output_path)

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  # 이 오디오는 끝났으니 상태를 ‘정지’로 바꾸자
        return stream
    
    def set_curr_state(self,audiotype, reinit):
        print('set_curr_state:',audiotype)
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1