import whisper
import librosa
import torch
import os
from datetime import datetime
import argparse

class VoiceToTextConverter:
    def __init__(self, model_size="base"):
        """
        Whisper 모델 초기화
        model_size: tiny, base, small, medium, large
        """
        # GPU 사용 가능 여부 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 가능한 디바이스: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU 메모리: {gpu_memory:.1f}GB")
            
            # large 모델 사용 시 메모리 경고
            if model_size == "large" and gpu_memory < 8:
                print("⚠️  경고: large 모델은 8GB 이상의 GPU 메모리를 권장합니다.")
                print("   메모리 부족 시 CPU 사용을 고려하세요.")
        else:
            print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")
        
        print(f"Whisper {model_size} 모델을 로딩 중...")
        
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print("모델 로딩 완료!")
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("GPU 메모리 부족으로 CPU로 전환합니다...")
                self.device = "cpu"
                self.model = whisper.load_model(model_size, device=self.device)
                print("CPU로 모델 로딩 완료!")
            else:
                raise e
    
    def transcribe_audio(self, audio_file_path, language="ko"):
        """
        librosa를 사용하여 오디오 파일을 텍스트로 변환
        """
        # 파일 존재 확인
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_file_path}")
        
        print(f"음성 파일 변환 중: {audio_file_path}")
        print(f"파일 크기: {os.path.getsize(audio_file_path)} bytes")
        
        try:
            # librosa로 오디오 파일 로드 (16kHz로 리샘플링)
            print("오디오 파일 로딩 중...")
            audio_data, sr = librosa.load(audio_file_path, sr=16000)
            print(f"오디오 로드 완료: 샘플레이트={sr}Hz, 길이={len(audio_data)/sr:.2f}초")
            
            # Whisper로 음성 인식 (numpy 배열 직접 전달)
            print("음성 인식 처리 중...")
            
            # NaN 값 방지를 위한 추가 옵션들
            transcribe_options = {
                "language": language,
                "verbose": True,
                "fp16": False,  # NaN 문제 해결을 위해 fp16 비활성화
                "temperature": 0.0,  # 온도를 0으로 설정하여 안정성 향상
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6
            }
            
            # 긴 오디오의 경우 청크 단위로 분할 처리
            if len(audio_data) > 16000 * 30:  # 30초 이상인 경우
                print("긴 오디오 파일입니다. 청크 단위로 처리합니다...")
                result = self._process_long_audio(audio_data, transcribe_options)
            else:
                result = self.model.transcribe(audio_data, **transcribe_options)
            
            print("✅ 음성 인식 완료!")
            return result
            
        except Exception as e:
            raise Exception(f"음성 인식 실패: {str(e)}")
    
    def _process_long_audio(self, audio_data, transcribe_options):
        """
        긴 오디오를 청크 단위로 분할하여 처리
        """
        import numpy as np
        
        chunk_length = 30  # 30초 청크
        sample_rate = 16000
        chunk_samples = chunk_length * sample_rate
        
        # 전체 결과를 저장할 구조
        full_result = {
            "text": "",
            "segments": [],
            "language": transcribe_options.get("language", "ko")
        }
        
        # 오디오를 청크로 분할
        total_chunks = len(audio_data) // chunk_samples + (1 if len(audio_data) % chunk_samples > 0 else 0)
        print(f"총 {total_chunks}개 청크로 분할하여 처리합니다...")
        
        current_time_offset = 0.0
        
        for i in range(total_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, len(audio_data))
            chunk_audio = audio_data[start_sample:end_sample]
            
            # 너무 짧은 청크는 건너뛰기
            if len(chunk_audio) < sample_rate * 0.5:  # 0.5초 미만
                continue
            
            print(f"청크 {i+1}/{total_chunks} 처리 중... ({start_sample/sample_rate:.1f}s - {end_sample/sample_rate:.1f}s)")
            
            try:
                # 각 청크 처리
                chunk_result = self.model.transcribe(chunk_audio, **transcribe_options)
                
                # 텍스트 결합
                if chunk_result["text"].strip():
                    full_result["text"] += chunk_result["text"] + " "
                
                # 세그먼트 시간 조정 후 추가
                for segment in chunk_result["segments"]:
                    adjusted_segment = segment.copy()
                    adjusted_segment["start"] += current_time_offset
                    adjusted_segment["end"] += current_time_offset
                    full_result["segments"].append(adjusted_segment)
                
            except Exception as e:
                print(f"청크 {i+1} 처리 중 오류 발생: {str(e)}")
                continue
            
            current_time_offset = end_sample / sample_rate
        
        # 텍스트 정리
        full_result["text"] = full_result["text"].strip()
        
        print(f"모든 청크 처리 완료! 총 텍스트 길이: {len(full_result['text'])}자")
        return full_result
    
    def save_transcript(self, result, output_file=None, base_filename=None):
        """
        변환된 텍스트를 파일로 저장
        """
        # src/text 디렉토리 생성
        text_dir = "src/text"
        os.makedirs(text_dir, exist_ok=True)
        
        if output_file is None:
            if base_filename:
                # 확장자 제거하고 txt로 변경
                filename = os.path.splitext(base_filename)[0] + ".txt"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"transcript_{timestamp}.txt"
            output_file = os.path.join(text_dir, filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 세그먼트별 상세 정보
            f.write("세그먼트별 상세:\n")
            f.write("-" * 30 + "\n")
            
            for segment in result["segments"]:
                start_time = self.format_time(segment["start"])
                end_time = self.format_time(segment["end"])
                f.write(f"[{start_time} - {end_time}] {segment['text']}\n")
        
        print(f"변환 결과가 저장되었습니다: {output_file}")
        return output_file
    
    def save_audio_file(self, uploaded_file):
        """
        업로드된 오디오 파일을 src/audio에 저장 (같은 이름으로)
        """
        # src/audio 디렉토리 생성
        audio_dir = "src/audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        # 업로드된 파일명 그대로 사용
        # base_filename = uploaded_file.name
        base_filename = datetime.now().strftime('%Y%m%d_%H%M%S')+'.'+os.path.splitext(uploaded_file.name)[1][1:]
        audio_path = os.path.join(audio_dir, base_filename)
        print(base_filename)
        # 파일 저장
        with open(audio_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        print(f"오디오 파일이 저장되었습니다: {audio_path}")
        return audio_path, base_filename
    
    def format_time(self, seconds):
        """
        초를 MM:SS 형식으로 변환
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def process_call_recording(self, audio_file, output_file=None, language="ko", base_filename=None):
        """
        통화 녹음 파일 전체 처리 과정
        """
        try:
            # 음성 인식 수행
            result = self.transcribe_audio(audio_file, language)
            
            # 결과 출력
            print("\n=== 변환 결과 ===")
            print(result["text"])
            
            # 파일로 저장 (같은 이름으로)
            saved_file = self.save_transcript(result, output_file, base_filename)
            
            return {
                "success": True,
                "text": result["text"],
                "segments": result["segments"],
                "output_file": saved_file,
                "base_filename": base_filename
            }
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

def main():
    parser = argparse.ArgumentParser(description="Whisper를 사용한 통화 녹음 음성인식")
    parser.add_argument("audio_file", help="변환할 오디오 파일 경로")
    parser.add_argument("-o", "--output", help="출력 텍스트 파일명")
    parser.add_argument("-m", "--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper 모델 크기 (기본값: base)")
    parser.add_argument("-l", "--language", default="ko", 
                       help="언어 코드 (기본값: ko)")
    
    args = parser.parse_args()
    
    # 음성인식 변환기 초기화
    converter = VoiceToTextConverter(model_size=args.model)
    
    # 통화 녹음 파일 처리
    result = converter.process_call_recording(
        audio_file=args.audio_file,
        output_file=args.output,
        language=args.language
    )
    
    if result["success"]:
        print(f"\n✅ 변환 완료!")
        print(f"📄 결과 파일: {result['output_file']}")
    else:
        print(f"\n❌ 변환 실패: {result['error']}")

if __name__ == "__main__":
    # 명령행 인수가 없으면 예제 실행
    import sys
    if len(sys.argv) == 1:
        print("사용법:")
        print("python practive_voice.py <오디오파일경로>")
        print("\n예시:")
        print("python practive_voice.py call_recording.wav")
        print("python practive_voice.py call_recording.mp3 -o result.txt -m small")
        
        # 테스트용 코드 (실제 파일이 있을 때만 실행)
        test_file = "test_audio.wav"
        if os.path.exists(test_file):
            converter = VoiceToTextConverter()
            converter.process_call_recording(test_file)
    else:
        main()
