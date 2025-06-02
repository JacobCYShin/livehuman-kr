'''
🎤 실시간 오디오 입력 (PCM, 16KHz)
         │
         ▼
🧠 LipASR.run_step()
    - 오디오 프레임 누적
    - Mel-spectrogram 계산
    - 16-frame 단위의 Mel chunk 리스트 생성
         │
         ▼
📤 feat_queue (ASR → Inference 연결)
         │
         ▼
🎨 inference()
    - 얼굴 이미지 & Mel chunk로 Wav2Lip 실행
    - 입 모양이 포함된 영상 프레임 생성
         │
         ▼
📺 process_frames()
    - 영상 프레임 위에 입 덮기
    - WebRTC로 비디오/오디오 푸시
'''

###############################################################################
# LiveTalking 프로젝트 - 음성 기반 입 모양 생성 (ASR to Mel)
###############################################################################

import time
import torch
import numpy as np

import queue
from queue import Queue
# import multiprocessing as mp  # 현재는 사용되지 않음

from baseasr import BaseASR  # 공통 ASR 처리 클래스
from wav2lip import audio    # Mel-spectrogram 추출 함수 포함

# BaseASR을 상속받아 Wav2Lip에 맞는 mel 특징 추출 기능 구현
class LipASR(BaseASR):

    # 한 프레임 단위로 음성 → Mel-spectrogram 특징 추출
    def run_step(self):
        ##############################################
        # 1. 오디오 프레임 수집 및 출력 큐 전달
        ##############################################
        for _ in range(self.batch_size * 2):
            frame, type, eventpoint = self.get_audio_frame()  # 20ms 오디오 수신
            self.frames.append(frame)                         # 내부 버퍼에 추가
            self.output_queue.put((frame, type, eventpoint))  # inference()로 전달

        ##############################################
        # 2. context 부족 시 특징 추출 스킵
        ##############################################
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return

        ##############################################
        # 3. Mel-spectrogram 생성
        ##############################################
        inputs = np.concatenate(self.frames)              # 연속적인 PCM 배열
        mel = audio.melspectrogram(inputs)                # (80, T) mel 스펙트로그램 생성

        ##############################################
        # 4. stride를 고려해 특징 자르기
        ##############################################
        left = max(0, self.stride_left_size * 80 / 50)    # left stride 위치 계산
        right = min(len(mel[0]), len(mel[0]) - self.stride_right_size * 80 / 50)

        mel_idx_multiplier = 80. * 2 / self.fps           # 프레임당 Mel 간격 (ex: 3.2)
        mel_step_size = 16                                # Wav2Lip에서 요구하는 step size

        i = 0
        mel_chunks = []
        while i < (len(self.frames) - self.stride_left_size - self.stride_right_size) / 2:
            start_idx = int(left + i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                # 끝에 도달하면 마지막 mel 범위 잘라서 넣기
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            else:
                mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        ##############################################
        # 5. 결과 mel_chunk 리스트를 inference queue로 전달
        ##############################################
        self.feat_queue.put(mel_chunks)

        ##############################################
        # 6. 오래된 프레임 삭제 (메모리 절약)
        ##############################################
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
