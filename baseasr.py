###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Apache License 2.0에 따라 사용이 허가됩니다.
#  이 파일은 라이선스에 명시된 조건 하에 사용 가능합니다.
#  라이선스 사본은 다음에서 확인할 수 있습니다:
#      http://www.apache.org/licenses/LICENSE-2.0
###############################################################################

import time
import numpy as np

import queue
from queue import Queue
import torch.multiprocessing as mp

from basereal import BaseReal  # 상위 시스템 또는 상태 관리용 베이스 클래스

# ASR (Automatic Speech Recognition, 음성 인식) 처리를 위한 기본 클래스 정의
class BaseASR:
    def __init__(self, opt, parent: BaseReal = None):
        self.opt = opt                  # 설정 옵션 객체
        self.parent = parent            # 상위 처리 시스템 객체 (BaseReal 타입)

        self.fps = opt.fps              # 초당 프레임 수 (예: 20이면 1초에 20프레임)
        self.sample_rate = 16000        # 오디오 샘플링 레이트 (16kHz)
        self.chunk = self.sample_rate // self.fps  # 프레임당 샘플 수 (예: 16000 / 50 = 320)

        self.queue = Queue()            # 입력 오디오 프레임을 담는 큐
        self.output_queue = mp.Queue()  # 처리된 오디오 출력을 위한 멀티프로세싱 큐

        self.batch_size = opt.batch_size  # 배치 사이즈 (사용 안 하고 있지만 설정됨)

        self.frames = []                # 누적된 오디오 프레임 리스트
        self.stride_left_size = opt.l   # 왼쪽 스트라이드 크기 (문맥 고려용)
        self.stride_right_size = opt.r  # 오른쪽 스트라이드 크기
        self.feat_queue = mp.Queue(2)   # 특징(feature) 벡터용 큐 (최대 크기 2)

    # 현재 저장된 오디오 입력 큐를 초기화(비우기)
    def flush_talk(self):
        self.queue.queue.clear()

    # 오디오 프레임을 입력 큐에 넣기 (16kHz, 20ms짜리 PCM 프레임)
    def put_audio_frame(self, audio_chunk, eventpoint=None):
        self.queue.put((audio_chunk, eventpoint))

    # 오디오 프레임을 큐에서 꺼내기
    # 반환값:
    # - frame: 오디오 데이터 (numpy array)
    # - type: 0 = 일반 음성, 1 = 무음 또는 사용자 정의 음성
    # - eventpoint: 오디오와 동기화되는 커스텀 이벤트
    def get_audio_frame(self):
        try:
            frame, eventpoint = self.queue.get(block=True, timeout=0.01)
            type = 0  # 일반 음성
        except queue.Empty:
            # 큐가 비어있고, 상위 상태가 1보다 크면 사용자 정의 음성을 가져옴
            if self.parent and self.parent.curr_state > 1:
                frame = self.parent.get_audio_stream(self.parent.curr_state)
                type = self.parent.curr_state  # 사용자 정의 상태 타입
            else:
                frame = np.zeros(self.chunk, dtype=np.float32)  # 무음 (제로 벡터)
                type = 1  # 무음 타입
            eventpoint = None

        return frame, type, eventpoint

    # 처리된 오디오 출력 프레임을 큐에서 가져옴
    # - 반환값: frame, type, eventpoint
    def get_audio_out(self): 
        return self.output_queue.get()

    # 스트라이드 크기만큼 프레임을 미리 받아서 warm-up 초기화
    # - 초기 프레임들을 output_queue에 넣고, 왼쪽 문맥 길이만큼 다시 제거
    def warm_up(self):
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame, type, eventpoint = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, type, eventpoint))

        # 왼쪽 문맥만큼 output 큐에서 제거
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    # ASR의 실제 실행 단위 (상속해서 구현 필요)
    def run_step(self):
        pass

    # 다음 특징 벡터(feature)를 가져옴
    # - block: True이면 blocking 방식으로 기다림
    # - timeout: 대기 시간 설정
    def get_next_feat(self, block, timeout):
        return self.feat_queue.get(block, timeout)
