import moviepy as mp
import speech_recognition as sr
from pathlib import Path
import argparse

def transcribe_video(video_path):
    """
    Transcribe speech from a video file and save the transcription.
    
    Args:
        video_path (str): Path to the video file
    """
    # Convert video path to Path object
    video_file = Path(video_path)
    
    # Extract audio from video
    print(f"Extracting audio from {video_file.name}...")
    video = mp.VideoFileClip(str(video_file))
    audio = video.audio
    
    # Save audio temporarily
    temp_audio = video_file.with_suffix('.wav')
    audio.write_audiofile(str(temp_audio))
    
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Read the audio file
    with sr.AudioFile(str(temp_audio)) as source:
        print("Processing audio...")
        # Record audio to memory
        audio_data = recognizer.record(source)
        
        # Attempt transcription
        try:
            print("Transcribing...")
            text = recognizer.recognize_google(audio_data)
            
            # Save transcription
            output_file = video_file.with_suffix('.txt')
            with open(output_file, 'w') as f:
                f.write(text)
            print(f"Transcription saved to {output_file}")
            
        except sr.UnknownValueError:
            print("Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")
    
    # Cleanup
    temp_audio.unlink()
    video.close()
    audio.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transcribe speech from a video file')
    parser.add_argument('video_path', help='Path to the video file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run transcription
    transcribe_video(args.video_path)

if __name__ == "__main__":
    main()