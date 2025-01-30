from openai import OpenAI
import argparse
import json
import os
import sys
import time
from typing import List, Dict
import re

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

    prompt = f"""You are an expert content analyzer focusing on viral potential. Analyze these clips:
{json.dumps(clips, indent=2)}

For each clip, evaluate using:

1. Audio Engagement (40% weight):
- Volume patterns and variations
- Voice intensity and emotional charge 
- Acoustic characteristics

2. Content Analysis (60% weight):
- Topic relevance and timeliness
- Controversial or debate-sparking elements
- "Quotable" phrases
- Discussion potential

For each clip, provide in this exact format:
1. **Clip Name: "[TITLE]"**
   Start: [START]s, End: [END]s
   Score: [1-10]
   Factors: [Key viral factors]
   Platforms: [Recommended platforms]

Rank clips by viral potential. Focus on measurable features in the data."""

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
    current_clip = {}
    lines = input_string.split('\n')
    
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        if re.match(r'^\d+\.\s\*\*Clip Name:', line):
            if current_clip:
                clips.append(current_clip)
                current_clip = {}
                
            name_match = re.search(r'Clip Name: "(.*?)"', line)
            if name_match:
                current_clip['name'] = name_match.group(1)
                
        elif 'Start:' in line and 'End:' in line:
            time_match = re.search(r'Start: ([\d.]+)s, End: ([\d.]+)s', line)
            if time_match:
                current_clip['start'] = float(time_match.group(1))
                current_clip['end'] = float(time_match.group(2))
                
        elif 'Score:' in line:
            score_match = re.search(r'Score: (\d+)', line)
            if score_match:
                current_clip['score'] = int(score_match.group(1))
                
        elif 'Factors:' in line:
            factors_match = re.search(r'Factors: (.+)', line)
            if factors_match:
                current_clip['factors'] = factors_match.group(1)
                
        elif 'Platforms:' in line:
            platforms_match = re.search(r'Platforms: (.+)', line)
            if platforms_match:
                current_clip['platforms'] = platforms_match.group(1)
    
    if current_clip:
        clips.append(current_clip)
    
    # Sort clips by score in descending order
    clips.sort(key=lambda x: x.get('score', 0), reverse=True)
    return clips

def save_top_clips_json(clips: List[Dict], output_file: str, num_clips: int = 20) -> None:
    top_clips = clips[:num_clips]
    output_data = {
        'top_clips': top_clips,
        'total_clips': len(clips),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Rank and extract top viral video clips metadata.')
    parser.add_argument('clips_json', help='JSON file containing clip information')
    parser.add_argument('--output_file', default='top_clips.json', help='Output JSON file for top clips')
    parser.add_argument('--site_url', default='http://localhost', help='Site URL for OpenRouter API')
    parser.add_argument('--site_name', default='Local Test', help='Site name for OpenRouter API')
    parser.add_argument('--num_clips', type=int, default=20, help='Number of top clips to extract')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise ValueError("Please set the OPEN_ROUTER_KEY environment variable")
        
        clips = load_clips(args.clips_json)
        ranked_results = rank_clips(clips, api_key, args.site_url, args.site_name)
        parsed_clips = parse_clip_data(ranked_results)
        
        save_top_clips_json(parsed_clips, args.output_file, args.num_clips)
        
        print(f"\nSuccessfully saved top {args.num_clips} clips to {args.output_file}")
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()