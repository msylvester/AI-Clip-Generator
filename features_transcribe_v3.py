import moviepy as mp
import speech_recognition as sr
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import json
import re

def load_transcription(input_path):
    """Load transcription data from a JSON file"""
    with open(input_path, 'r') as f:
        return json.load(f)

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

class TextSegment:
    def __init__(self, text, emphasized=False):
        self.text = text
        self.emphasized = emphasized

def parse_text_with_emphasis(text):
    """Parse text with *emphasis* markers into segments"""
    segments = []
    pattern = r'\*(.*?)\*|([^\*]+)'
    
    for match in re.finditer(pattern, text):
        emphasized_text, normal_text = match.groups()
        if emphasized_text:
            segments.append(TextSegment(emphasized_text, emphasized=True))
        elif normal_text:
            segments.append(TextSegment(normal_text, emphasized=False))
    
    return segments

def create_text_image(text_data, size, font_settings):
    """Create a PIL image with text in a bounded box, supporting emphasis"""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Load regular and bold fonts
    try:
        regular_font = ImageFont.truetype(font_settings['font_path'], font_settings['font_size'])
        bold_font = ImageFont.truetype(font_settings['bold_font_path'], font_settings['font_size'])
    except:
        print("Could not load custom fonts, using default")
        regular_font = bold_font = ImageFont.load_default()
    
    # Parse text segments if it's a string, or use pre-parsed segments
    if isinstance(text_data, str):
        segments = parse_text_with_emphasis(text_data)
    else:
        segments = text_data
    
    # Calculate text wrapping
    box_width = int(size[0] * font_settings['width_percent'])
    text = ''.join(segment.text for segment in segments)
    avg_char_width = draw.textlength('A', font=regular_font)
    chars_per_line = int(box_width / avg_char_width)
    
    # Pre-process text wrapping
    current_line = []
    lines = []
    current_length = 0
    
    for segment in segments:
        words = segment.text.split()
        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 <= chars_per_line:
                current_line.append((word, segment.emphasized))
                current_length += word_length + 1
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [(word, segment.emphasized)]
                current_length = word_length
    
    if current_line:
        lines.append(current_line)
    
    # Render text
    line_height = font_settings['font_size'] + font_settings['line_spacing']
    total_height = len(lines) * line_height
    start_y = size[1] - total_height - font_settings['bottom_padding']
    
    for line_idx, line in enumerate(lines):
        current_x = (size[0] - sum(draw.textlength(word + ' ', 
                                                 font=bold_font if emphasized else regular_font)
                                 for word, emphasized in line)) // 2
        y = start_y + (line_idx * line_height)
        
        for word, emphasized in line:
            font = bold_font if emphasized else regular_font
            
            # Draw outline
            if font_settings['outline_width'] > 0:
                for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    draw.text(
                        (current_x + dx * font_settings['outline_width'],
                         y + dy * font_settings['outline_width']),
                        word + ' ',
                        font=font,
                        fill=font_settings['outline_color']
                    )
            
            # Draw text
            draw.text(
                (current_x, y),
                word + ' ',
                fill=font_settings['font_color'],
                font=font
            )
            
            current_x += draw.textlength(word + ' ', font=font)
    
    return np.array(img)

def create_subtitle_clips(words_with_timestamps, video_size, font_settings):
    """Create subtitle clips from transcribed text with emphasis support"""
    subtitle_clips = []
    
    for start, end, text in words_with_timestamps:
        text_image = create_text_image(text, (video_size[0], video_size[1]), font_settings)
        text_clip = (mp.ImageClip(text_image)
                    .with_start(float(start))
                    .with_duration(float(end) - float(start)))
        subtitle_clips.append(text_clip)
    
    return subtitle_clips

def process_video(video_path, font_settings, generate_transcription=True, output_path=None):
    """Process video to add transcribed text overlay with emphasis support"""
    video_file = Path(video_path)
    if output_path is None:
        output_path = video_file.with_suffix('.subtitled.mp4')
    
    transcription_path = video_file.with_suffix('.transcription.json')
    
    print(f"Processing {video_file.name}...")
    video = mp.VideoFileClip(str(video_file))
    
    if generate_transcription:
        print("Generating new transcription...")
        recognizer = sr.Recognizer()
        words_with_timestamps = transcribe_with_timestamps(recognizer, video.audio)
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

def main():
    parser = argparse.ArgumentParser(description='Add transcribed text overlay to video with emphasis support')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--output', help='Output path (optional)')
    parser.add_argument('--generate-transcription', action='store_true',
                      help='Generate new transcription (if false, will use existing transcription file)')
    
    # Font customization arguments
    parser.add_argument('--font-path', default="/Library/Fonts/Arial.ttf",
                      help='Path to regular font file (TTF format)')
    parser.add_argument('--bold-font-path', default="/Library/Fonts/Arial Bold.ttf",
                      help='Path to bold font file (TTF format)')
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
        'bold_font_path': args.bold_font_path,
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