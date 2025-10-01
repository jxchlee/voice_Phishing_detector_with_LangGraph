# 보이스피싱 탐지 시스템

음성 파일을 업로드하여 텍스트로 변환하고, AI를 통해 보이스피싱 여부를 판단하는 Streamlit 애플리케이션입니다.

## 기능

- 🎤 음성 파일을 텍스트로 변환 (Whisper 모델 사용)
- 🔍 AI 기반 보이스피싱 탐지 (LangGraph + OpenAI GPT 사용)
- 📊 위험도 분석 및 상세 결과 제공
- 🎵 지원 음성 형식: WAV, MP3, M4A, FLAC, OGG

## 설치 및 실행

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:
`webpage/model/.env` 파일에 다음 API 키들이 설정되어 있는지 확인:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

3. Streamlit 앱 실행:
```bash
streamlit run voice_phishing_detector.py
```

## 사용법

1. 웹 브라우저에서 앱이 열리면 음성 파일을 업로드합니다.
2. 사이드바에서 Whisper 모델 크기와 언어를 선택합니다.
3. "보이스피싱 분석 시작" 버튼을 클릭합니다.
4. 분석 결과를 확인합니다.

## 주의사항

- 처음 실행 시 Whisper 모델 다운로드로 인해 시간이 걸릴 수 있습니다.
- 큰 모델일수록 정확도가 높지만 처리 시간이 오래 걸립니다.
- 인터넷 연결이 필요합니다 (OpenAI API 및 Tavily 검색 사용).