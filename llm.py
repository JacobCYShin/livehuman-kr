import time
import os
import requests
from logger import logger
from basereal import BaseReal

def llm_response(message, nerfreal: BaseReal):
    start = time.perf_counter()
    logger.info("llm Time init: local HTTP server ready")

    # POST 요청에 보낼 JSON 데이터
    payload = {
        "content": message
    }

    try:
        # 로컬 서버로 요청
        response = requests.post(
            "http://localhost:8888/generate_response/",
            json=payload,
            timeout=60
        )

        end = time.perf_counter()
        logger.info(f"llm Time to response: {end - start:.2f}s")

        # 응답 텍스트 가져오기
        if response.status_code == 200:
            full_response = response.json().get("response", "")
        else:
            logger.error(f"LLM HTTP Error: {response.status_code}")
            full_response = "[ERROR] 로컬 LLM 응답 실패"

    except Exception as e:
        logger.exception("LLM 서버 호출 중 예외 발생")
        full_response = "[EXCEPTION] 로컬 LLM 예외 발생"

    # 응답을 문장 단위로 분할하여 순차 전달
    buffer = ""
    for i, char in enumerate(full_response):
        buffer += char
        if char in ",.!;:，。！？：；" and len(buffer.strip()) > 10:
            logger.info(buffer.strip())
            nerfreal.put_msg_txt(buffer.strip())
            buffer = ""

    # 마지막 남은 텍스트 처리
    if buffer.strip():
        nerfreal.put_msg_txt(buffer.strip())

    logger.info("llm Time complete.")
