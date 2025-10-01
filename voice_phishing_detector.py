import streamlit as st
import os
import tempfile
from webpage.practive_voice import VoiceToTextConverter
from webpage.model.graph_model import analyze_voice_phishing

def main():
    st.set_page_config(
        page_title="보이스피싱 탐지 시스템",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 보이스피싱 탐지 시스템")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("설정")
    model_size = st.sidebar.selectbox(
        "Whisper 모델 크기",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="모델이 클수록 정확도가 높지만 처리 시간이 오래 걸립니다."
    )
    
    language = st.sidebar.selectbox(
        "언어 설정",
        ["ko", "en", "auto"],
        index=0,
        help="음성 인식할 언어를 선택하세요."
    )
    
    # 메인 컨텐츠
    st.header("📁 음성 파일 업로드")
    uploaded_file = st.file_uploader(
        "음성 파일을 선택하세요",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
        help="지원 형식: WAV, MP3, M4A, FLAC, OGG"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"파일 업로드 완료: {uploaded_file.name}")
            st.info(f"파일 크기: {uploaded_file.size:,} bytes")
            
        with col2:
            analyze_button = st.button("🚀 보이스피싱 분석 시작", type="primary", use_container_width=True)
        
        # 오디오 플레이어
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
        
        # 분석 실행
        if analyze_button:
            analyze_audio(uploaded_file, model_size, language)
    else:
        st.info("음성 파일을 업로드해주세요.")

def analyze_audio(uploaded_file, model_size, language):
    """음성 파일 분석 함수"""
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. 임시 파일로 저장
        status_text.text("📁 파일 준비 중...")
        progress_bar.progress(10)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # 2. 음성 인식 모델 초기화
        status_text.text("🤖 AI 모델 로딩 중...")
        progress_bar.progress(30)
        
        converter = VoiceToTextConverter(model_size=model_size)
        
        # 3. 음성을 텍스트로 변환
        status_text.text("🎤 음성 인식 처리 중...")
        progress_bar.progress(60)
        result = converter.transcribe_audio(temp_file_path, language=language)
        transcribed_text = result["text"]
        
        # 4. 보이스피싱 분석
        status_text.text("🔍 보이스피싱 분석 중...")
        progress_bar.progress(80)
        
        phishing_result = analyze_voice_phishing(transcribed_text)
        
        # 5. 결과 표시
        status_text.text("✅ 분석 완료!")
        progress_bar.progress(100)
        
        # 결과 섹션 - 전체 화면에 표시
        st.markdown("---")
        
        # 보이스피싱 분석 결과
        st.subheader("🚨 보이스피싱 분석 결과")
        
        # 위험도 판단 (간단한 키워드 기반)

        risk_score = int(phishing_result.splitlines()[0].split(":")[1])
        if risk_score >= 7:
            st.error("⚠️ **높은 위험도** - 보이스피싱 가능성이 높습니다!")
        elif risk_score >= 4:
            st.warning("⚡ **중간 위험도** - 주의가 필요합니다.")
        else:
            st.success("✅ **낮은 위험도** - 정상적인 통화로 보입니다.")
        
        # AI 분석 결과
        with st.expander("AI 상세 분석 결과", expanded=True):
            st.markdown(phishing_result)
        
        # 음성 인식 결과 버튼들
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 변환된 텍스트 보기", use_container_width=True):
                st.session_state.show_text = True
                st.session_state.show_segments = False
        
        with col2:
            if st.button("🎯 세그먼트별 상세 보기", use_container_width=True):
                st.session_state.show_segments = True
                st.session_state.show_text = False
        
        # 텍스트 표시
        if st.session_state.get('show_text', False):
            st.subheader("📝 변환된 텍스트")
            st.text_area("전체 텍스트", transcribed_text, height=300, key="text_display")
        
        # 세그먼트 표시
        if st.session_state.get('show_segments', False) and "segments" in result:
            st.subheader("🎯 세그먼트별 상세 내용")
            for i, segment in enumerate(result["segments"]):
                start_time = format_time(segment["start"])
                end_time = format_time(segment["end"])
                st.write(f"**[{start_time} - {end_time}]** {segment['text']}")
        
        # 임시 파일 정리
        os.unlink(temp_file_path)
        
    except Exception as e:
        st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
        status_text.text("❌ 분석 실패")
        progress_bar.progress(0)

def format_time(seconds):
    """초를 MM:SS 형식으로 변환"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    main()