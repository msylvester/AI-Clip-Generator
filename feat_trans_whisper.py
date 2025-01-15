import moviepy as mp
import speech_recognition as sr
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import json
import whisper
from moviepy import VideoFileClip
import torch
def create_text_image(text, size, font_settings):
    """Create a PIL image with text in a bounded box"""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_settings['font_path'], font_settings['font_size'])
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_settings['font_size'])
        except:
            print("Could not load custom font, using default")
            font = ImageFont.load_default()
    
    box_width = int(size[0] * font_settings['width_percent'])
    avg_char_width = draw.textlength('A', font=font)
    chars_per_line = int(box_width / avg_char_width)
    wrapped_text = textwrap.fill(text, width=chars_per_line)
    lines = wrapped_text.split('\n')

    line_height = font_settings['font_size'] + font_settings['line_spacing']
    total_height = len(lines) * line_height
    start_y = size[1] - total_height - font_settings['bottom_padding']
    
    for i, line in enumerate(lines):
        line_width = draw.textlength(line, font=font)
        x = (size[0] - line_width) // 2
        y = start_y + (i * line_height)

        if font_settings['outline_width'] > 0:
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.text(
                    (x + dx * font_settings['outline_width'], 
                     y + dy * font_settings['outline_width']),
                    line,
                    font=font,
                    fill=font_settings['outline_color']
                )
        
        draw.text((x, y), line, fill=font_settings['font_color'], font=font)
    
    return np.array(img)

def transcribe_with_timestamps(model, audio_source):
    """Get transcription with timestamps using Whisper"""
    print("Generating timestamped transcription...")
    words_with_timestamps = []
    
    # Save audio to temporary file
    temp_audio = "temp_audio.wav"
    audio_source.write_audiofile(temp_audio, logger=None)
    
    # Run Whisper transcription
    result = model.transcribe(temp_audio, language='en')
    
    # Process segments
    for segment in result["segments"]:
        words_with_timestamps.append((
            segment["start"],
            segment["end"],
            segment["text"].upper()  # Keep consistent with current uppercase format
        ))
    
    # Clean up temp file
    Path(temp_audio).unlink(missing_ok=True)
    
    return words_with_timestamps
def save_transcription(words_with_timestamps, output_path):
    """Save transcription data to a JSON file"""
    with open(output_path, 'w') as f:
        json.dump(words_with_timestamps, f, indent=2)
    print(f"Transcription saved to {output_path}")

def load_transcription(input_path):
    """Load transcription data from a JSON file"""
    with open(input_path, 'r') as f:
        return json.load(f)

def create_subtitle_clips(words_with_timestamps, video_size, font_settings):
    """Create subtitle clips from transcribed text"""
    subtitle_clips = []
    
    for start, end, text in words_with_timestamps:
        text_image = create_text_image(text, (video_size[0], video_size[1]), font_settings)
        text_clip = (mp.ImageClip(text_image)
                    .with_start(float(start))  # Ensure start time is float
                    .with_duration(float(end) - float(start)))  # Ensure duration is float
        subtitle_clips.append(text_clip)
    
    return subtitle_clips

def process_video(video_path, font_settings, generate_transcription=True, output_path=None):
    """Process video to add transcribed text overlay"""
    video_file = Path(video_path)
    if output_path is None:
        output_path = video_file.with_suffix('.subtitled.mp4')
    
    transcription_path = video_file.with_suffix('.transcription.json')
    
    print(f"Processing {video_file.name}...")
    video = mp.VideoFileClip(str(video_file))
    
    if generate_transcription:
        print("Loading Whisper model...")
        model = whisper.load_model("base")  # Can use "tiny", "base", "small", "medium", "large"
        
        print("Generating new transcription...")
        words_with_timestamps = transcribe_with_timestamps(model, video.audio)
        save_transcription(words_with_timestamps, transcription_path)
    else:
        print("Loading existing transcription...")
        if not transcription_path.exists():
            raise FileNotFoundError(f"Transcription file not found: {transcription_path}")
        words_with_timestamps = load_transcription(transcription_path)
    
    print("Creating subtitle clips...")
    subtitle_clips = create_subtitle_clips(words_with_timestamps, video.size, font_settings)
    
    print("Adding subtitles to video...")
    final_video = mp.CompositeVideoClip([video] + subtitle_clips)
    
    print(f"Writing output to {output_path}...")
    final_video.write_videofile(str(output_path), 
                              codec='libx264',
                              audio_codec='aac')
    
    video.close()
    final_video.close()
    print("Done!")

def parse_color(color_str):
    """Convert color string to RGB tuple"""
    try:
        if color_str.startswith('#'):
            color_str = color_str[1:]
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
        color_map = {
            'red': (255, 0, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'yellow': (255, 255, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0)
        }
        return color_map.get(color_str.lower(), (255, 255, 255))
    except:
        return (255, 255, 255)

def main():
    parser = argparse.ArgumentParser(description='Add transcribed text overlay to video')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output', help='Output path (optional)')
    parser.add_argument('--generate-transcription', action='store_true',
                      help='Generate new transcription (if false, will use existing transcription file)')
    
    # Font customization arguments
    parser.add_argument('--font-path', default="/Library/Fonts/Arial.ttf",
                      help='Path to font file (TTF format)')
    parser.add_argument('--font-size', type=int, default=30,
                      help='Font size in pixels')
    parser.add_argument('--font-color', default='white',
                      help='Font color (hex code or name)')
    parser.add_argument('--outline-color', default='black',
                      help='Outline color (hex code or name)')
    parser.add_argument('--outline-width', type=int, default=2,
                      help='Width of text outline in pixels')
    parser.add_argument('--line-spacing', type=int, default=4,
                      help='Spacing between lines in pixels')
    parser.add_argument('--bottom-padding', type=int, default=50,
                      help='Padding from bottom of screen in pixels')
    parser.add_argument('--width-percent', type=float, default=0.8,
                      help='Width of text box as percentage of video width (0.0-1.0)')
    
    args = parser.parse_args()

    font_settings = {
        'font_path': args.font_path,
        'font_size': args.font_size,
        'font_color': parse_color(args.font_color),
        'outline_color': parse_color(args.outline_color),
        'outline_width': args.outline_width,
        'line_spacing': args.line_spacing,
        'bottom_padding': args.bottom_padding,
        'width_percent': args.width_percent
    }
    
    process_video(args.video_path, font_settings, 
                 generate_transcription=args.generate_transcription,
                 output_path=args.output)

if __name__ == "__main__":
    main()