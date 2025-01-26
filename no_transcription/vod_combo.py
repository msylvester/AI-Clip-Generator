from moviepy import VideoFileClip
from openai import OpenAI
import argparse
import json
import os
import sys
import time
from typing import List, Dict, Tuple
import re
def extract_clip(input_file: str, output_file: str, start_time: float, end_time: float) -> None:
    try:
        with VideoFileClip(input_file) as video:
            clip = video.subclipped(start_time, end_time)
            clip.write_videofile(output_file, codec='libx264')
    except Exception as e:
        raise RuntimeError(f"Failed to extract clip: {str(e)}")

def parse_timestamp(timestamp: str) -> float:
    parts = timestamp.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError("Time must be in HH:MM:SS format")

def load_clips(json_path: str) -> List[Dict]:
    try:
        with open(json_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Clips file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_path}")

def rank_clips(clips: List[Dict], api_key: str, site_url: str = "", site_name: str = "") -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }
    )

  prompt = f"""Please rank these clips based on their potential virality:
{json.dumps(clips, indent=2)}

For each clip provided (and ONLY these clips), give:
1. Clip Name, Start time and end time
2. A brief explanation of virality potential based on length and content type

Keep explanations focused and concise."""

    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that ranks video clips. Keep explanations brief and focused on virality potential."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=1000
    )
    
    return completion.choices[0].message.content

def parse_clip_data(input_string: str) -> list[dict]:
    clips = []
    lines = input_string.split('\n')
    print(f'the lines are {lines} ')
    for i in range(len(lines)):
        line = lines[i]
        if re.match(r'^\d+\.\s\*\*Clip Name:', line):
            name_match = re.search(r'Clip Name: "(.*?)"', line)
            time_match = re.search(r'Start: ([\d.]+)s, End: ([\d.]+)s', lines[i + 1])
            
            if name_match and time_match:
                clips.append({
                    'name': name_match.group(1),
                    'start': time_match.group(1),
                    'end': time_match.group(2)
                })
    
    return clips
def process_ranked_clips(input_file: str, ranked_clips: str, max_clips: int = 5) -> List[Tuple[str, float, float]]:
    """Extract top N clip information from ranking results"""
    clips_to_extract = []
    current_clip = None
    start_time = None
    print(f'the ranked clips are {ranked_clips}')
    for line in ranked_clips.split('\n'):
        if "Start time" in line and "end time" in line:
            try:
                times = line.split("Start time")[1].split("end time")
                start = times[0].strip().strip(':,')
                end = times[1].strip().strip(':,')
                if current_clip:
                    clips_to_extract.append((
                        f"{current_clip}_{start}-{end}.mp4",
                        parse_timestamp(start),
                        parse_timestamp(end)
                    ))
            except Exception:
                continue
        elif line.strip() and not line.startswith(('1.', '2.', '-')):
            current_clip = line.strip()
    
    return clips_to_extract

def main():
    parser = argparse.ArgumentParser(description='Extract and rank viral video clips.')
    parser.add_argument('input_file', help='Input video file path')
    parser.add_argument('clips_json', help='JSON file containing clip information')
    parser.add_argument('--output_dir', default='top_five_clips', help='Output directory for extracted clips')
    parser.add_argument('--site_url', default='http://localhost', help='Site URL for OpenRouter API')
    parser.add_argument('--site_name', default='Local Test', help='Site name for OpenRouter API')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Get API key from environment
        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise ValueError("Please set the OPEN_ROUTER_KEY environment variable")
        
        # Load and rank clips
        clips = load_clips(args.clips_json)
        ranked_results = rank_clips(clips, api_key, args.site_url, args.site_name)
        
        # print("\nRanking Results:")
        # print("-" * 50)
        # print(f'rnaked results {parse_clip_data(ranked_results)}')
        # Process and extract clips
        clips_to_extract = process_ranked_clips(args.input_file, ranked_results, max_clips=5)
        #print(f'about to extract {clips_to_extract}')
       
        # Parse the ranked results string into a list of clips
      #  parsed_clips = parse_clip_data(ranked_results)

        # Extract the first 5 clips
        # Parse the ranked results string into a list of clips
        parsed_clips = parse_clip_data(ranked_results)
        print(f'the ranked results are {parsed_clips}')
        # Extract the first 5 clips
        for i, clip in enumerate(parsed_clips[:5]):
            print('extracting')
            output_path = os.path.join(args.output_dir, f"{clip['name']}.mp4")
         #   print(f"\nExtracting clip: {clip['name']}")
            extract_clip(args.input_file, output_path, float(clip['start']), float(clip['end']))
        # for i, clip in enumerate(ranked_results[:5]):
        #     print('extracting')
        #     output_path = os.path.join(args.output_dir, clip['name'])
        #     print(f"\nExtracting clip: {clip['name']}")
        #     extract_clip(args.input_file, output_path, float(clip['start']), float(clip['end']))
            
        # for i, (output_name, start, end) in enumerate(parse_clip_data(ranked_results)):
        #     if i >= 5:  # Stop after processing 5 items
        #         break
        #     print('extracting')
        #     output_path = os.path.join(args.output_dir, output_name)
        #     print(f"\nExtracting clip: {output_name}")
        #     extract_clip(args.input_file, output_path, start, end)
        # # for output_name, start, end in parse_clip_data(ranked_results):
        #     print('extracting')
        #     output_path = os.path.join(args.output_dir, output_name)
        #     print(f"\nExtracting clip: {output_name}")
        #     extract_clip(args.input_file, output_path, start, end)
            
        print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()