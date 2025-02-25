import moviepy as mp
import speech_recognition as sr
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap

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
    
    # Define text box size (adjustable width percentage)
    box_width = int(size[0] * font_settings['width_percent'])
       
    # Wrap text to fit box width
    avg_char_width = draw.textlength('A', font=font)
    chars_per_line = int(box_width / avg_char_width)
    wrapped_text = textwrap.fill(text, width=chars_per_line)
    lines = wrapped_text.split('\n')

    # Calculate total height needed
    line_height = font_settings['font_size'] + font_settings['line_spacing']
    total_height = len(lines) * line_height
    
    # Calculate starting Y position
    start_y = size[1] - total_height - font_settings['bottom_padding']
    
    # Draw each line centered
    for i, line in enumerate(lines):
        line_width = draw.textlength(line, font=font)
        x = (size[0] - line_width) // 2
        y = start_y + (i * line_height)

        # Draw outline if enabled
        if font_settings['outline_width'] > 0:
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.text(
                    (x + dx * font_settings['outline_width'], 
                     y + dy * font_settings['outline_width']),
                    line,
                    font=font,
                    fill=font_settings['outline_color']
                )
        
        # Draw main text
        draw.text((x, y), line, fill=font_settings['font_color'], font=font)
    
    return np.array(img)

def transcribe_with_timestamps(recognizer, audio_source):
    """Get transcription with timestamps using small word chunks"""
    print("Generating timestamped transcription...")
    words_with_timestamps = []
    
    # Reduce chunk duration for more frequent timestamps
    chunk_duration = 2  # seconds
    word_limit = 5  # maximum words per segment
    offset = 0
    
    while offset < audio_source.duration:
        audio_chunk = audio_source.subclipped(offset, min(offset + chunk_duration, audio_source.duration))
        chunk_wav = Path(f"temp_chunk_{offset}.wav")
        audio_chunk.write_audiofile(str(chunk_wav),  logger=None)
        
        with sr.AudioFile(str(chunk_wav)) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                words = text.upper().split()
                
                # Split text into smaller segments of max 5 words
                for i in range(0, len(words), word_limit):
                    segment_words = words[i:i + word_limit]
                    segment_text = ' '.join(segment_words)
                    
                    # Calculate sub-timings within the chunk
                    segment_duration = chunk_duration * (len(segment_words) / len(words))
                    segment_start = offset + (chunk_duration * (i / len(words)))
                    segment_end = segment_start + segment_duration
                    
                    words_with_timestamps.append((
                        segment_start,
                        segment_end,
                        segment_text
                    ))
                    
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"API Error: {e}")
                
        chunk_wav.unlink()
        offset += chunk_duration
    
    # Clean up overlapping segments
    words_with_timestamps.sort(key=lambda x: x[0])
    cleaned_timestamps = []
    
    for i, (start, end, text) in enumerate(words_with_timestamps):
        if i > 0:
            # Adjust timing to prevent overlap
            prev_end = cleaned_timestamps[-1][1]
            if start < prev_end:
                start = prev_end
        if start < end:  # Only add valid segments
            cleaned_timestamps.append((start, end, text))
    
    return cleaned_timestamps
def create_subtitle_clips(words_with_timestamps, video_size, font_settings):
    """Create subtitle clips from transcribed text"""
    subtitle_clips = []
    
    for start, end, text in words_with_timestamps:
        text_image = create_text_image(text, (video_size[0], video_size[1]), font_settings)
        text_clip = (mp.ImageClip(text_image)
                    .with_start(start)
                    .with_duration(end - start))
        subtitle_clips.append(text_clip)
    
    return subtitle_clips

def process_video(video_path, font_settings, output_path=None):
    """Process video to add transcribed text overlay"""
    video_file = Path(video_path)
    if output_path is None:
        output_path = video_file.with_suffix('.subtitled.mp4')
    
    print(f"Processing {video_file.name}...")

    video = mp.VideoFileClip(str(video_file))
    audio = video.audio
    
    recognizer = sr.Recognizer()
    words_with_timestamps = transcribe_with_timestamps(recognizer, audio)
    print(f'the words with timestamps are {words_with_timestamps}')
    subtitle_clips = create_subtitle_clips(words_with_timestamps, video.size, font_settings)
    
    print("Adding subtitles to video...")
    final_video = mp.CompositeVideoClip([video] + subtitle_clips)
    
    print(f"Writing output to {output_path}...")
    final_video.write_videofile(str(output_path), 
                              codec='libx264',
                              audio_codec='aac')
    
    video.close()
    audio.close()
    final_video.close()
    print("Done!")

def parse_color(color_str):
    """Convert color string to RGB tuple"""
    try:
        # Handle hex colors
        if color_str.startswith('#'):
            color_str = color_str[1:]
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
        # Handle named colors
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
        return (255, 255, 255)  # Default to white

def main():
    parser = argparse.ArgumentParser(description='Add transcribed text overlay to video')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output', help='Output path (optional)')
    
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
    
    # Create font settings dictionary
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
    
    process_video(args.video_path, font_settings, args.output)

if __name__ == "__main__":
    main()