# 🎤 실시간 AI 립싱크 스트리밍 (한국어 버전)

본 프로젝트는 원작자의 WebRTC 기반 디지털 휴먼 스트리밍 시스템을 한국어 환경에 맞게 최적화한 버전입니다.
`app.py`를 실행하면 **한국어 음성 입력에 최적화된 립싱크 디지털 휴먼 스트리밍**을 직접 테스트할 수 있습니다.

---

## ⚡ 빠른 실행 가이드 (Quickstart)

```bash
conda activate nerfstream
chcp 65001                # (Windows에서 UTF-8 인코딩 설정)
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip_taeri
```

> 실행 후, 브라우저에서 아래 주소로 접속:
>
> **[http://localhost:8010/dashboard.html](http://localhost:8010/dashboard.html)**

필수 파일 목록:

* `dashboard.html`: 사용자 인터페이스
* `client.js`: 브라우저 음성 입력 및 WebRTC 제어
* `srs.sdk.js`: SRS WebRTC 미디어 서버 SDK

---

## 🧑‍🎤 현재 사용 중인 모델 구성

| 항목       | 설명                                                     |
| -------- | ------------------------------------------------------ |
| 립싱크 모델   | `Wav2Lip` 256 버전 (`wav2lip/models/wav2lip_v4.py`) 사용 중 |
| TTS 모델   | `ESPnet` 기반 한국어 TTS (MeloTTS는 Windows 설치 이슈로 제외됨)      |
| LLM      | `polyglot-ko-1.3b-chat` (GPU 메모리 이슈로 제한적으로 사용)         |
| 모델 파일 위치 | `models/wav2lip.pth` 경로에 저장 필요                         |

✅ Wav2Lip 구조 수정 시:
`wav2lip/models/__init__.py`에서 아래 라인을 수정하여 사용할 모델 버전을 명시합니다:

```python
from .wav2lip_v4 import Wav2Lip, Wav2Lip_disc_qual  # 256 모델 사용 시
# 또는
from .wav2lip_v3 import Wav2Lip, Wav2Lip_disc_qual  # 512 모델 사용 시
```

---

## 🔊 TTS 사용 방식

로컬에서 실행 중인 TTS 서버(ESPnet 기반)를 다음과 같이 호출합니다:

```python
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
        res = requests.post("http://localhost:7009/api/tts", json=payload)
```

* TTS 서버 실행 명령: `uvicorn app.main:app --host 0.0.0.0 --port 7009`

---

## 🤖 LLM 연동 방식
* Polyglot-Ko 모델 로컬 실행: `uvicorn app:app --host 0.0.0.0 --port 8888`
* WebRTC 시스템은 내부적으로 해당 LLM을 호출하여 응답 생성을 처리합니다.

---

## 🛠 성능 측정 및 최적화 예정 작업 (ToDo)

* [ ] **MeloTTS 연동**: Windows 설치 대응 및 모델 로딩 시간 최적화
* [ ] **LLM 연동 개선**: polyglot-ko 외 경량 모델 도입 및 예외 처리 강화
* [ ] **Wav2Lip 고속화**: TorchCompile 또는 ONNX/TensorRT 변환 테스트 및 비교 분석
* [ ] **입술 마스킹 정밀화**: 얼굴 전체 대신 입술 영역만 합성하도록 모델 입출력 수정
---

## 📂 디렉토리 구조 (일부)

```
├── app.py                  # 실행 진입점
├── ttsreal.py              # LocalTTS 정의 (포트 7009 호출)
├── webrtc.py               # WebRTC 시그널링 처리
├── llm.py                  # 디지털 휴먼용 LLM 응답 처리
├── lipreal.py              # Wav2Lip 립싱크 처리 (BaseReal 상속)
├── lipasr.py               # ASR 처리 (BaseASR 상속)
├── baseasr.py              # ASR 공통 로직
├── basereal.py             # 디지털 휴먼 공통 처리 클래스
├── web/
│   ├── dashboard.html      # 클라이언트 대시보드
│   ├── client.js           # 브라우저 입력 및 제어
│   ├── srs.sdk.js          # SRS SDK
├── models/                 # 립싱크, 음성 모델 체크포인트
├── utils/                  # 디버깅/리사이즈/좌표 변환 스크립트 등
```

---

## 🧾 coords.pkl (얼굴 합성 좌표) 생성 방법

디지털 아바타의 얼굴 합성 좌표는 `.npy` → `.pkl`로 변환해야 하며, 구조는 다음과 같습니다:

* 리스트 형식: `List[Tuple[np.int64]]`
* 순서: `(y1, y2, x1, x2)`


## 🧩 시스템 구성도

```
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
```
| 구성요소              | 역할                               |
| ----------------- | -------------------------------- |
| Client            | WebRTC 대시보드 or API 요청자           |
| aiohttp           | WebRTC offer 처리 및 REST API 제공    |
| build\_nerfreal() | 모델/아바타 로딩 및 디지털 휴먼 인스턴스 생성       |
| BaseReal 계열       | LipReal, MuseReal 등 디지털 휴먼 로직 수행 |
| ASR, TTS          | 오디오 분석 및 합성 (입력 텍스트 → 음성)        |
| inference()       | Wav2Lip 등 입 모양 생성 (음성 + 얼굴)      |
| render()          | 오디오/비디오 WebRTC 트랙으로 푸시           |
| HumanPlayer       | aiortc용 오디오/비디오 track 생성기        |
| WebRTC PC         | 브라우저와 미디어 트랙 송수신                 |

---

## 📦 주요 특징

* **한국어 TTS/ASR 지원 (모든 주요 스크립트에 한글 주석 포함)**
* **실시간 WebRTC 기반 스트리밍**
* MuseTalk, Wav2Lip, Ultralight 디지털 휴먼 모델 호환
* 타이핑한 한국어 텍스트 → 음성 → 립싱크 비디오 출력
* 브라우저 기반 대화형 UI 지원
## 🙏 원작자에게 감사의 말씀

본 코드는 [원작 GitHub 리포지토리](https://https://github.com/lipku/LiveTalking)를 기반으로, 한국어 환경 및 TTS/LLM 연동 최적화를 수행한 포크 버전입니다. 원작자의 노력에 깊이 감사드립니다.

