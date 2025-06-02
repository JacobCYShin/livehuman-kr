###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku
#  Licensed under the Apache License, Version 2.0
###############################################################################

# 표준 라이브러리
import asyncio
import json
import logging
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union

# FFmpeg 라이브러리의 프레임/패킷 처리 모듈
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

# 오디오/비디오 처리에 사용될 시간 단위 설정
AUDIO_PTIME = 0.020  # 20ms 단위 오디오 패킷 처리 주기
VIDEO_CLOCK_RATE = 90000  # 비디오용 타임스탬프 단위 (90kHz)
VIDEO_PTIME = 1 / 25       # 25fps 기준 프레임 간 간격
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

# aiortc: WebRTC 관련 라이브러리
from aiortc import MediaStreamTrack

# 로그 초기화
logging.basicConfig()
logger = logging.getLogger(__name__)
from logger import logger as mylogger  # 사용자 정의 logger

# WebRTC를 위한 커스텀 트랙 정의 (비디오 또는 오디오)
class PlayerStreamTrack(MediaStreamTrack):
    """
    비디오 또는 오디오 스트림을 전달하는 WebRTC 트랙 클래스
    """

    def __init__(self, player, kind):
        super().__init__()  # MediaStreamTrack 초기화
        self.kind = kind  # 'video' 또는 'audio'
        self._player = player  # 미디어 데이터 제공 객체
        self._queue = asyncio.Queue()  # 스트리밍 프레임을 저장할 큐
        self.timelist = []  # 시간 동기화를 위한 타임스탬프 리스트

        if self.kind == 'video':
            self.framecount = 0  # 프레임 카운트
            self.lasttime = time.perf_counter()  # 마지막 프레임 수신 시간
            self.totaltime = 0  # 누적 시간

    # 내부 상태 변수들
    _start: float
    _timestamp: int

    # 다음 프레임의 타임스탬프 계산 (비디오 또는 오디오)
    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        # 비디오의 경우
        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                # 이전 타임스탬프에 프레임 시간만큼 더함
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)  # 동기화를 위한 sleep
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('video start:%f', self._start)
            return self._timestamp, VIDEO_TIME_BASE

        # 오디오의 경우
        else:
            if hasattr(self, "_timestamp"):
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                wait = self._start + (self._timestamp / SAMPLE_RATE) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('audio start:%f', self._start)
            return self._timestamp, AUDIO_TIME_BASE

    # 다음 프레임을 수신하고 timestamp를 부여
    async def recv(self) -> Union[Frame, Packet]:
        self._player._start(self)  # 플레이어에 현재 트랙 시작을 알림

        # 큐에서 (프레임, 이벤트포인트) 형태로 가져옴
        frame, eventpoint = await self._queue.get()

        # 타임스탬프 계산 후 프레임에 설정
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base

        # 이벤트 포인트가 있다면 알림 전송
        if eventpoint:
            self._player.notify(eventpoint)

        # None인 경우 종료 처리
        if frame is None:
            self.stop()
            raise Exception

        # FPS 측정용 로깅 (비디오만 해당)
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime = 0

        return frame

    # 트랙 종료 처리
    def stop(self):
        super().stop()
        if self._player is not None:
            self._player._stop(self)  # 플레이어에게 종료 알림
            self._player = None

# 플레이어가 동작하는 워커 스레드 함수 정의
def player_worker_thread(
    quit_event,       # 종료 신호를 위한 이벤트 객체
    loop,             # asyncio 이벤트 루프
    container,        # 실질적으로 프레임을 생성하는 객체 (NerfReal 또는 BaseReal 등)
    audio_track,      # 오디오 스트림 트랙
    video_track       # 비디오 스트림 트랙
):
    container.render(quit_event, loop, audio_track, video_track)
    # container는 디지털 휴먼 구현체로 추정되며, render() 메서드를 통해 오디오/비디오 프레임을 푸시함

# 실제 오디오/비디오 프레임을 aiortc로 전송하기 위한 미디어 플레이어 클래스
class HumanPlayer:

    def __init__(
        self, nerfreal, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        self.__thread: Optional[threading.Thread] = None  # 백그라운드 스레드
        self.__thread_quit: Optional[threading.Event] = None  # 종료 시그널 이벤트

        # 스트림 트랙이 시작되었는지 추적
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        # 오디오/비디오 트랙 초기화 (PlayerStreamTrack은 앞서 정의됨)
        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")

        # 실제 렌더링(프레임 생성)을 담당할 객체 (보통 NerfReal 또는 BaseReal)
        self.__container = nerfreal

    # 디지털 휴먼 쪽으로 이벤트 전달
    def notify(self, eventpoint):
        self.__container.notify(eventpoint)

    # 외부에서 audio 트랙에 접근할 수 있도록 하는 프로퍼티
    @property
    def audio(self) -> MediaStreamTrack:
        """
        오디오 트랙 반환 (aiortc용 MediaStreamTrack)
        """
        return self.__audio

    # 외부에서 video 트랙에 접근할 수 있도록 하는 프로퍼티
    @property
    def video(self) -> MediaStreamTrack:
        """
        비디오 트랙 반환 (aiortc용 MediaStreamTrack)
        """
        return self.__video

    # 특정 트랙이 시작될 때 호출 (처음에만 워커 스레드를 생성해서 render() 실행)
    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()  # 스레드 종료 신호
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,  # 위에서 정의한 render 호출 함수
                args=(
                    self.__thread_quit,
                    asyncio.get_event_loop(),  # 현재 asyncio 이벤트 루프
                    self.__container,
                    self.__audio,
                    self.__video
                ),
            )
            self.__thread.start()

    # 특정 트랙이 종료될 때 호출
    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        # 모든 트랙이 종료되었고, 스레드가 살아 있으면 종료
        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        # 컨테이너도 정리
        if not self.__started and self.__container is not None:
            self.__container = None

    # 내부 로그 출력 함수
    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)
