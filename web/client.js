// WebRTC PeerConnection 객체
var pc = null;

// WebRTC offer 생성 및 서버 연결 절차
function negotiate() {
    // 서버로부터 video, audio 스트림을 "받기만(recvonly)" 하도록 설정
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    // 브라우저 측 offer 생성 → setLocalDescription
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // ICE candidate 수집이 끝날 때까지 대기
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        // 서버에 offer 정보 전송
        var offer = pc.localDescription;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();  // 서버에서 answer + sessionid 수신
    }).then((answer) => {
        document.getElementById('sessionid').value = answer.sessionid;  // 세션 ID 저장
        return pc.setRemoteDescription(answer);  // answer 설정
    }).catch((e) => {
        alert(e);  // 오류 발생 시 알림
    });
}

// WebRTC 연결 시작 함수
function start() {
    var config = {
        sdpSemantics: 'unified-plan'  // 최신 SDP 방식 사용
    };

    // STUN 서버 사용 여부 체크박스 값에 따라 ICE 서버 설정
    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    // PeerConnection 객체 생성
    pc = new RTCPeerConnection(config);

    // 서버로부터 들어오는 track을 처리 (비디오 / 오디오 스트림 연결)
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            document.getElementById('video').srcObject = evt.streams[0];
        } else {
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    // UI 버튼 토글: Start → Stop
    document.getElementById('start').style.display = 'none';
    negotiate();  // offer 생성 및 연결 시도
    document.getElementById('stop').style.display = 'inline-block';
}

// 연결 종료 함수
function stop() {
    document.getElementById('stop').style.display = 'none';

    // 500ms 후 PeerConnection 닫기
    setTimeout(() => {
        pc.close();
    }, 500);
}

// 페이지를 떠날 때 자동 종료
window.onunload = function(event) {
    setTimeout(() => {
        pc.close();
    }, 500);
};

// 페이지 닫기 전 사용자에게 확인 요청 + 연결 종료
window.onbeforeunload = function (e) {
    setTimeout(() => {
        pc.close();
    }, 500);

    e = e || window.event;
    if (e) {
        e.returnValue = '정말 닫으시겠습니까?';
    }
    return '정말 닫으시겠습니까?';
};
