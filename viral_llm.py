import moviepy as mp
import speech_recognition as sr
import tempfile
import os
from pydub import AudioSegment
from llama_cpp import Llama
from moviepy import VideoFileClip

def test_pipeline(video_path, llm, test_duration=30):
    """Test the analysis pipeline on first 30 seconds"""
    try:
        print(f"Testing pipeline with {test_duration} second clip...")
        
        # Create temporary shortened video
        clip = VideoFileClip(video_path).subclipped(0, test_duration)
        temp_path = "temp_test_video.mp4"
        clip.write_videofile(temp_path, audio_codec='aac')
        
        # Run analysis on short clip
        test_results = analyze_video_virality(temp_path, llm)
        
        if test_results:
            print("\nTest successful! Found these moments in sample:")
            for i, (start, end, score) in enumerate(test_results[:2], 1):
                print(f"Test Clip {i}: {format_timestamp(start)} - {format_timestamp(end)}, Score: {score:.4f}")
            return True
        else:
            print("Test completed but no viral moments detected. Check if this is expected.")
            return True
            
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False
    finally:
        # Cleanup
        if os.path.exists("temp_test_video.mp4"):
            os.remove("temp_test_video.mp4")
            
def analyze_video_virality(video_path, llm, segment_duration=10):
    """
    Analyzes a video file to identify potentially viral moments with progress tracking
    """
    try:
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Load video
        video = mp.VideoFileClip(video_path)
        total_duration = video.duration
        
        # Calculate total segments for progress tracking
        total_segments = int(total_duration / segment_duration) + 1
        processed_segments = 0
        
        # Extract audio and save temporarily
        print("Extracting audio...")
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
        
        # Initialize speech recognizer
        r = sr.Recognizer()
        
        # Split audio into segments
        audio = AudioSegment.from_wav(temp_audio_path)
        duration_ms = len(audio)
        segment_duration_ms = segment_duration * 1000
        
        viral_segments = []
        
        print("\nProcessing video segments:")
        for start_ms in range(0, duration_ms, segment_duration_ms):
            end_ms = min(start_ms + segment_duration_ms, duration_ms)
            segment = audio[start_ms:end_ms]
            
            # Update progress
            processed_segments += 1
            progress = (processed_segments / total_segments) * 100
            print(f"\rProgress: {progress:.1f}% (Segment {processed_segments}/{total_segments})", end="")
            
            # Process segment
            temp_segment = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_segment_path = temp_segment.name
            temp_segment.close()
            segment.export(temp_segment_path, format="wav")
            
            # Transcribe segment
            with sr.AudioFile(temp_segment_path) as source:
                audio_data = r.record(source)
            
            try:
                text = r.recognize_google(audio_data)
                if text:
                    virality_score = llm_analyze_virality(llm, text)
                    viral_segments.append((start_ms/1000, end_ms/1000, virality_score))
            except sr.UnknownValueError:
                print(f"\nSpeech recognition could not understand audio at {start_ms/1000:.1f}-{end_ms/1000:.1f} seconds")
            except sr.RequestError as e:
                print(f"\nCould not request results from speech recognition service at {start_ms/1000:.1f}-{end_ms/1000:.1f} seconds; {e}")
            
            os.unlink(temp_segment_path)
        
        print("\nProcessing complete!")
        
        # Clean up
        os.unlink(temp_audio_path)
        video.close()
        
        # Sort and validate segments
        if not viral_segments:
            print("Warning: No viral segments detected")
            return []
            
        viral_segments.sort(key=lambda x: x[2], reverse=True)
        return viral_segments
        
    except Exception as e:
        print(f"\nError during video analysis: {str(e)}")
        return []

# Rest of your existing code remains the same
def initialize_llm(model_path):
    return Llama(model_path=model_path, n_ctx=2048, n_batch=512)

def llm_analyze_virality(llm, text):
    prompt = f"""
    Analyze the following text from a video and rate its potential virality on a scale from 0 to 1, 
    where 0 is not viral at all and 1 is extremely viral. 

    Consider factors like:
    - Emotional impact
    - Uniqueness
    - Relevance to current trends
    - Potential for sharing

    Provide only the numerical score as output.

    Text: {text}

    Virality Score:
    """

    response = llm(prompt, max_tokens=20, stop=["Text:", "\n"], echo=False)
    try:
        score = float(response['choices'][0]['text'].strip())
        return max(0, min(score, 1))
    except ValueError:
        print(f"Failed to parse LLM output: {response['choices'][0]['text']}")
        return 0

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    video_path = "dec_18_fiv_min_cip.mp4"
    model_path = "./llm_dec_3/llama-2-7b-chat.Q4_K_M.gguf"
    
    llm = initialize_llm(model_path)
    
    # First run test pipeline
    print("Running test pipeline...")
    if test_pipeline(video_path, llm, test_duration=30):
        print("\nTest successful! Proceeding with full video analysis...\n")
        
        # Run full analysis
        viral_moments = analyze_video_virality(video_path, llm)
        
        if viral_moments:
            print("\nTop potentially viral moments:")
            for i, (start, end, score) in enumerate(viral_moments[:5], 1):
                print(f"\nClip {i}")
                print(f"Timestamp: {format_timestamp(start)} - {format_timestamp(end)}")
                print(f"Virality Score: {score:.4f}")
        else:
            print("\nNo viral moments were detected in the video.")
    else:
        print("\nTest pipeline failed. Please check the errors above before processing the full video.")