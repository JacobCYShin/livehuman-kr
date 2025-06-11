import time
import os
from basereal import BaseReal         # 디지털 휴먼 시스템의 컨트롤러 클래스
from logger import logger             # 로그 출력 모듈

# LLM 응답 생성 함수
def llm_response(message, nerfreal: BaseReal):
    start = time.perf_counter()  # 처리 시간 측정 시작

    from openai import OpenAI  # OpenAI 클라이언트 (DashScope 호환용 사용)
    client = OpenAI(
        # OpenAI API 키는 환경변수에서 가져옴 (DASHSCOPE_API_KEY)
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # DashScope에서 제공하는 OpenAI 호환 API 엔드포인트
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    end = time.perf_counter()
    logger.info(f"llm Time init: {end-start}s")  # API 초기화 시간 로그

    # 채팅 요청 생성 (스트리밍 모드)
    completion = client.chat.completions.create(
        model="qwen-plus",  # 사용할 모델 이름
        messages=[          # 대화 히스토리 구성
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': message}
        ],
        stream=True,        # 스트리밍 방식으로 응답을 받음
        stream_options={"include_usage": True}  # 토큰 사용량도 마지막에 포함
    )

    result = ""    # 누적된 전체 메시지
    first = True   # 첫 응답 여부 체크

    # 스트리밍 응답 처리
    for chunk in completion:
        if len(chunk.choices) > 0:
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start}s")  # 첫 응답까지 걸린 시간
                first = False

            msg = chunk.choices[0].delta.content  # 현재 chunk에서 받은 텍스트 조각
            lastpos = 0

            # 문장 구분 기호를 기준으로 메시지를 끊어서 처리
            for i, char in enumerate(msg):
                if char in ",.!;:，。！？：；":  # 문장 구분 기호 리스트
                    result = result + msg[lastpos:i+1]  # 문장 끝까지 누적
                    lastpos = i + 1

                    # 누적된 문장이 어느 정도 길면 바로 TTS에 전달
                    if len(result) > 10:
                        logger.info(result)
                        nerfreal.put_msg_txt(result)  # 실시간으로 TTS에 전달
                        result = ""  # 버퍼 초기화

            result = result + msg[lastpos:]  # 남은 문자 추가

    end = time.perf_counter()
    logger.info(f"llm Time to last chunk: {end-start}s")  # 전체 응답 완료까지 시간 측정

    nerfreal.put_msg_txt(result)  # 마지막 남은 문장도 TTS로 전달
