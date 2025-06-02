import logging  # 파이썬 표준 로깅 모듈

# 로그 기록기를 생성합니다. 모듈 이름을 기준으로 logger 인스턴스를 만듭니다.
logger = logging.getLogger(__name__)

# 로그의 최소 출력 레벨을 DEBUG로 설정 (즉, DEBUG 이상은 모두 처리)
logger.setLevel(logging.DEBUG)

# 로그 출력 형식을 지정합니다: 시간 - 모듈이름 - 로그레벨 - 메시지
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 로그를 파일로 기록할 핸들러 생성
fhandler = logging.FileHandler('livetalking.log')  # 로그를 'livetalking.log' 파일에 저장

# 위에서 만든 포맷터를 파일 핸들러에 연결
fhandler.setFormatter(formatter)

# 파일에 기록할 로그의 레벨은 INFO 이상으로 제한
fhandler.setLevel(logging.INFO)

# 파일 핸들러를 logger에 등록
logger.addHandler(fhandler)

# 콘솔 출력용 핸들러도 추가할 수 있지만, 현재는 주석 처리됨
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(sformatter)
# logger.addHandler(handler)
