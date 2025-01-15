import moviepy as mp
import tempfile
import os
from pathlib import Path
import json
import whisper
from moviepy import VideoFileClip
import tiktoken
import requests

def count_tokens(text):
    """
    Count tokens in text using tiktoken encoder for GPT models.
    While this won't be exact for DeepSeek, it provides a good approximation.
    """
    try:
        encoder = tiktoken.encoding_for_model("gpt-4")
        token_count = len(encoder.encode(text))
        return token_count
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return 0

def test_pipeline(video_path, api_key, site_url, site_name, test_duration=30):
    """Test the analysis pipeline on first 30 seconds"""
    try:
        print(f"Testing pipeline with {test_duration} second clip...")
        
        clip = VideoFileClip(video_path).subclipped(0, test_duration)
        temp_path = "temp_test_video.mp4"
        clip.write_videofile(temp_path, audio_codec='aac')
        
        test_results = analyze_video_virality(temp_path, api_key, site_url, site_name)
        
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
        if os.path.exists("temp_test_video.mp4"):
            os.remove("temp_test_video.mp4")

def analyze_video_virality(video_path, api_key, site_url, site_name, segment_duration=10, analysis_mode="chunked"):
    """
    Analyzes a video file to identify potentially viral moments.
    """
    if analysis_mode == "single":
        return analyze_video_single_block(video_path, api_key, site_url, site_name, segment_duration)
    else:
        return analyze_video_chunked(video_path, api_key, site_url, site_name, segment_duration)

def transcribe_with_whisper(video_path):
    """Transcribe video using Whisper model"""
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    print("Transcribing audio...")
    result = model.transcribe(video_path)
    
    return result["segments"]

def analyze_video_chunked(video_path, api_key, site_url, site_name, segment_duration=10):
    """Analyzes video by breaking it into chunks using Whisper"""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        print("Starting chunked analysis...")
        segments = transcribe_with_whisper(video_path)
        
        viral_segments = []
        total_segments = len(segments)
        
        print("\nProcessing video segments:")
        for idx, segment in enumerate(segments, 1):
            progress = (idx / total_segments) * 100
            print(f"\rProgress: {progress:.1f}% (Segment {idx}/{total_segments})", end="")
            
            text = segment["text"]
            start_time = segment["start"]
            end_time = segment["end"]
            
            if text.strip():
                virality_score = analyze_virality_openrouter(text, api_key, site_url, site_name)
                viral_segments.append((start_time, end_time, virality_score))
        
        print("\nProcessing complete!")
        
        if not viral_segments:
            print("Warning: No viral segments detected")
            return []
            
        viral_segments.sort(key=lambda x: x[2], reverse=True)
        return viral_segments
        
    except Exception as e:
        print(f"\nError during video analysis: {str(e)}")
        return []

def analyze_video_single_block(video_path, api_key, site_url, site_name, clip_duration=10):
    """Analyzes the entire video as a single block using Whisper"""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        print("Loading video...")
        video = VideoFileClip(video_path)
        total_duration = video.duration
        
        print("Starting transcription...")
        segments = transcribe_with_whisper(video_path)
        
        # Combine all transcribed text
        full_text = " ".join(segment["text"] for segment in segments)
        
        if not full_text.strip():
            print("No speech detected in video")
            return []
        
        print("Analyzing entire transcript for viral content...")
        virality_score = analyze_virality_openrouter(full_text, api_key, site_url, site_name)
        
        # Find the segment with the highest density of speech
        max_segment = max(segments, key=lambda x: len(x["text"]) / (x["end"] - x["start"]))
        start_time = max_segment["start"]
        end_time = min(start_time + clip_duration, total_duration)
        
        return [(start_time, end_time, virality_score)]
            
    except Exception as e:
        print(f"Error during video analysis: {str(e)}")
        return []
    finally:
        if 'video' in locals():
            video.close()

def analyze_virality_openrouter(text, api_key, site_url, site_name, max_context_length=65536):
    """Analyze text virality using OpenRouter API"""
    try:
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

        char_count = len(prompt)
        estimated_tokens = char_count / 4
        print(f"The number of tokens are {estimated_tokens} tokens")

        if estimated_tokens > max_context_length:
            print(f"Warning: Input may exceed context window (est. {estimated_tokens:.0f} tokens > {max_context_length})")
            ratio = max_context_length / estimated_tokens
            truncated_length = int(len(text) * (ratio * 0.8))
            text = text[:truncated_length] + "..."
            prompt = prompt.replace(text, text[:truncated_length] + "...")
            print(f"Text truncated to approximately {count_tokens(prompt)} tokens")

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
            data=json.dumps({
                "model": "deepseek/deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        response_data = response.json()
        score_text = response_data['choices'][0]['message']['content'].strip()
        
        try:
            score = float(score_text)
            return max(0, min(score, 1))
        except ValueError:
            print(f"Failed to parse API output: {score_text}")
            return 0
            
    except Exception as e:
        print(f"Error calling OpenRouter API: {str(e)}")
        return 0

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    video_path = "two_min_joker.mp4"  # Replace with your video path
    api_key = os.getenv("OPEN_ROUTER_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_KEY environment variable not found")
    site_url = "<YOUR_SITE_URL>"      # Replace with your site URL
    site_name = "<YOUR_SITE_NAME>"    # Replace with your site name
    
    # Allow choosing analysis mode
    analysis_mode = input("Choose analysis mode ('chunked' or 'single'): ").lower()
    if analysis_mode not in ["chunked", "single"]:
        print("Invalid mode selected. Defaulting to 'chunked'")
        analysis_mode = "chunked"
    
    if analysis_mode == "chunked":
        # Run test pipeline first for chunked mode
        print("Running test pipeline...")
        if test_pipeline(video_path, api_key, site_url, site_name, test_duration=30):
            print("\nTest successful! Proceeding with full video analysis...\n")
            
            viral_moments = analyze_video_virality(video_path, api_key, site_url, site_name, 
                                                 analysis_mode="chunked")
            
            if viral_moments:
                print("\nTop potentially viral moments:")
                for i, (start, end, score) in enumerate(viral_moments[:5], 1):
                    print(f"\nClip {i}")
                    print(f"Timestamp: {format_timestamp(start)} - {format_timestamp(end)}")
                    print(f"Virality Score: {score:.4f}")
            else:
                print("\nNo viral moments were detected in the video.")
    else:
        print("Analyzing video for most viral clip...")
        viral_moments = analyze_video_virality(video_path, api_key, site_url, site_name, 
                                             analysis_mode="single")
        
        if viral_moments:
            start, end, score = viral_moments[0]
            print("\nMost viral clip found:")
            print(f"Timestamp: {format_timestamp(start)} - {format_timestamp(end)}")
            print(f"Virality Score: {score:.4f}")
        else:
            print("\nNo viral moments were detected in the video.")