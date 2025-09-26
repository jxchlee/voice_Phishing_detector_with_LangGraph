# 1. 베이스 이미지 (base_image에 있는 베이스 이미지)
FROM base_image:latest

# 2. 작업 디렉터리 생성 및 설정
WORKDIR . /app

# 3. 로컬 파일을 컨테이너 내부로 복사
COPY . ./

# 4. 필요 패키지 설치


# 5. 컨테이너 실행 시 실행될 명령어
CMD ["streamlit", "run", "streamlit_voice_app.py"]

EXPOSE 8501