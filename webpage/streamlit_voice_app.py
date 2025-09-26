import streamlit as st
import tempfile
import os
from datetime import datetime
import pandas as pd

# torch 관련 경고 억제
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from practive_voice import VoiceToTextConverter
from model.graph_model import analyze_voice_phishing

# 페이지 설정
st.set_page_config(
    page_title="보이스피싱 탐지 시스템",
    page_icon="🛡️",
    layout="wide"
)

# 제목
st.title("🛡️ 보이스피싱 탐지 시스템")
st.markdown("음성 파일을 업로드하면 텍스트로 변환하고 보이스피싱 여부를 분석해드립니다.")

# 사이드바 설정
st.sidebar.header("설정")
model_size = st.sidebar.selectbox(
    "Whisper 모델 크기",
    ["tiny", "base", "small", "medium", "large"],
    index=1  # base가 기본값
)

language = st.sidebar.selectbox(
    "언어",
    ["ko", "en", "auto"],
    index=0  # 한국어가 기본값
)

# 메인 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 파일 업로드")
    uploaded_file = st.file_uploader(
        "음성 파일을 선택하세요",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
        help="지원 형식: WAV, MP3, M4A, FLAC, OGG"
    )
    
    if uploaded_file is not None:
        st.success(f"파일 업로드 완료: {uploaded_file.name}")
        st.info(f"파일 크기: {uploaded_file.size:,} bytes")
        
        # 오디오 플레이어
        st.audio(uploaded_file, format='audio/wav')

with col2:
    st.header("⚙️ 처리 상태")
    if uploaded_file is not None:
        if st.button("🚀 분석 시작", type="primary"):
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 음성 변환기 초기화 (캐싱으로 성능 향상)
                @st.cache_resource
                def load_model(model_size):
                    return VoiceToTextConverter(model_size=model_size)
                
                converter = load_model(model_size)
                
                status_text.text("오디오 파일 저장 중...")
                progress_bar.progress(10)
                
                # 오디오 파일을 src/audio에 저장 (같은 이름으로)
                audio_path, base_filename = converter.save_audio_file(uploaded_file)
                
                status_text.text("음성 파일 분석 중...")
                progress_bar.progress(30)
                
                # 음성 인식 수행
                result = converter.process_call_recording(
                    audio_path, 
                    language=language if language != "auto" else None,
                    base_filename=base_filename
                )
                
                progress_bar.progress(70)
                
                if result["success"]:
                    status_text.text("보이스피싱 분석 중...")
                    
                    # 보이스피싱 분석 수행
                    try:
                        phishing_analysis = analyze_voice_phishing(result["text"])
                        result["phishing_analysis"] = phishing_analysis
                        progress_bar.progress(100)
                        status_text.text("✅ 분석 완료!")
                    except Exception as e:
                        st.warning(f"보이스피싱 분석 중 오류 발생: {str(e)}")
                        result["phishing_analysis"] = "보이스피싱 분석을 수행할 수 없습니다."
                        progress_bar.progress(100)
                        status_text.text("✅ 텍스트 변환 완료 (보이스피싱 분석 실패)")
                    
                    # 세션 상태에 결과 저장
                    st.session_state.result = result
                    st.session_state.processed = True
                    st.rerun()
                else:
                    st.error(f"변환 실패: {result['error']}")
                    
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                progress_bar.empty()
                status_text.empty()

# 결과 표시
if hasattr(st.session_state, 'processed') and st.session_state.processed:
    st.header("📊 분석 결과")
    
    result = st.session_state.result
    
    # 보이스피싱 분석 결과를 상단에 표시
    if "phishing_analysis" in result:
        st.subheader("🛡️ 보이스피싱 분석 결과")
        
        # 분석 결과에 따라 색상 구분
        analysis_text = result["phishing_analysis"]
        if "보이스피싱" in analysis_text and ("의심" in analysis_text or "위험" in analysis_text):
            st.error("⚠️ 보이스피싱 의심")
            st.error(analysis_text)
        elif "정상" in analysis_text or "안전" in analysis_text:
            st.success("✅ 정상 통화")
            st.success(analysis_text)
        else:
            st.info("📋 분석 결과")
            st.info(analysis_text)
        
        st.divider()
    
    # 전체 텍스트
    st.subheader("📝 변환된 텍스트")
    st.text_area("", value=result["text"], height=150, disabled=True)
    
    # 다운로드 버튼
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.download_button(
            label="📄 텍스트 다운로드",
            data=result["text"],
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        # CSV 형태로 세그먼트 데이터 준비
        segments_data = []
        for segment in result["segments"]:
            segments_data.append({
                "시작시간": f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}",
                "종료시간": f"{int(segment['end']//60):02d}:{int(segment['end']%60):02d}",
                "텍스트": segment["text"]
            })
        
        df = pd.DataFrame(segments_data)
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        
        st.download_button(
            label="📊 CSV 다운로드",
            data=csv,
            file_name=f"segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # 세그먼트별 상세 정보 (접을 수 있는 형태로 변경)
    with st.expander("🎯 세그먼트별 상세 보기", expanded=False):
        # 필터링 옵션
        col1, col2 = st.columns([1, 3])
        with col1:
            min_duration = st.slider("최소 길이 (초)", 0.0, 10.0, 0.0, 0.1)
        
        # 세그먼트 표시
        filtered_segments = [
            seg for seg in result["segments"] 
            if (seg["end"] - seg["start"]) >= min_duration
        ]
        
        st.info(f"총 {len(filtered_segments)}개 세그먼트 (필터링 후)")
        
        # 세그먼트를 더 컴팩트한 형태로 표시
        for i, segment in enumerate(filtered_segments):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
                end_time = f"{int(segment['end']//60):02d}:{int(segment['end']%60):02d}"
                duration = segment['end'] - segment['start']
                st.write(f"**{start_time} - {end_time}**")
                st.caption(f"({duration:.1f}초)")
            
            with col2:
                st.write(segment["text"])
            
            if i < len(filtered_segments) - 1:  # 마지막이 아닌 경우에만 구분선
                st.divider()
    
    # 통계 정보
    st.subheader("📊 분석 통계")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_duration = max([seg["end"] for seg in result["segments"]])
        st.metric("총 길이", f"{int(total_duration//60):02d}:{int(total_duration%60):02d}")
    
    with col2:
        st.metric("총 세그먼트", len(result["segments"]))
    
    with col3:
        avg_duration = sum([seg["end"] - seg["start"] for seg in result["segments"]]) / len(result["segments"])
        st.metric("평균 세그먼트 길이", f"{avg_duration:.1f}초")
    
    with col4:
        total_chars = len(result["text"])
        st.metric("총 글자 수", f"{total_chars:,}자")
    


# 푸터
st.markdown("---")
st.markdown("🔧 **Powered by OpenAI Whisper, LangGraph & Streamlit**")

# 초기화 버튼
if st.sidebar.button("🔄 초기화"):
    if hasattr(st.session_state, 'processed'):
        del st.session_state.processed
    if hasattr(st.session_state, 'result'):
        del st.session_state.result
    st.rerun()