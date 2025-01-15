import moviepy as mp
import speech_recognition as sr
import tempfile
import os
from pydub import AudioSegment
import requests
import json
import os
from moviepy import VideoFileClip
import tiktoken


'''

(1)chunkking 
(2)fine tune the prompting (across diff types of content --> were primarliy of focus)
(3) 

'''

def count_tokens(text):
    """
    Count tokens in text using tiktoken encoder for GPT models.
    While this won't be exact for DeepSeek, it provides a good approximation.
    """
    try:
        # Initialize the encoder (using GPT-4 encoding as approximation)
        encoder = tiktoken.encoding_for_model("gpt-4")
        token_count = len(encoder.encode(text))
        return token_count
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return 0

#DeepSeek V3 has 64k context
#

#bigger the context, the the worse teh performance (consider chunking)
#use a tokenizer to see how many tokens will be processed


#look for max context of the model on openrouter
#64k


#the OAI tokenizer has a web interface to see how many tokens are in your text, doesnt mean DeepSeek would have the same amount though. They are dividing words in different ways usually






def test_pipeline(video_path, api_key, site_url, site_name, test_duration=30):
    """Test the analysis pipeline on first 30 seconds"""
    try:
        print(f"Testing pipeline with {test_duration} second clip...")
        
        # Create temporary shortened video
        clip = VideoFileClip(video_path).subclipped(0, test_duration)
        temp_path = "temp_test_video.mp4"
        clip.write_videofile(temp_path, audio_codec='aac')
        
        # Run analysis on short clip
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
        # Cleanup
        if os.path.exists("temp_test_video.mp4"):
            os.remove("temp_test_video.mp4")
            
def analyze_video_virality(video_path, api_key, site_url, site_name, segment_duration=10):
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
                    virality_score = analyze_virality_openrouter(text, api_key, site_url, site_name)
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

def analyze_virality_openrouter(text, api_key, site_url, site_name, max_context_length=65536):  # 64k context window
    """
    Analyze text virality using OpenRouter API with token count check
    """
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

        # Check context length before making API call
        # Using character count as a rough approximation since DeepSeek's exact tokenization differs
        # Average of 4 characters per token as a conservative estimate
        char_count = len(prompt)
        estimated_tokens = char_count / 4
        print(f"The number of tokens are {estimated_tokens} tokens")

        if estimated_tokens > max_context_length:
            print(f"Warning: Input may exceed context window (est. {estimated_tokens:.0f} tokens > {max_context_length})")
            # Truncate text to roughly fit within context window
            # This is a simple truncation - you might want to implement smarter chunking
            ratio = max_context_length / estimated_tokens
            truncated_length = int(len(text) * (ratio * 0.8))  # 0.8 as safety factor
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
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    video_path = "two_min_joker.mp4"
    api_key = os.getenv("OPEN_ROUTER_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_KEY environment variable not found")
    site_url = "<YOUR_SITE_URL>"      # Replace with your site URL
    site_name = "<YOUR_SITE_NAME>"    # Replace with your site name
    
    # First run test pipeline
    print("Running test pipeline...")
    if test_pipeline(video_path, api_key, site_url, site_name, test_duration=30):
        print("\nTest successful! Proceeding with full video analysis...\n")
        
        # Run full analysis
        viral_moments = analyze_video_virality(video_path, api_key, site_url, site_name)
        
        if viral_moments:
            print("\nTop potentially viral moments:")
            for i, (start, end, score) in enumerate(viral_moments[:5], 1):
                print(f"\nClip {i}")
                print(f"Timestamp: {format_timestamp(start)} - {format_timestamp(end)}")
                print(f"Virality Score: {score:.4f}")
        else:
            print("\nNo viral moments were detected in the video.")