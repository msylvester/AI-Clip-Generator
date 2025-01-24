from openai import OpenAI
import json
import os
from typing import List, Dict
import time

def load_clips(json_path: str) -> List[Dict]:
    """Load clips from a JSON file."""
    try:
        with open(json_path, 'r') as file:
            clips = json.load(file)
        return clips
    except FileNotFoundError:
        raise FileNotFoundError(f"Clips file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_path}")

def rank_clips(clips: List[Dict], api_key: str, site_url: str = "", site_name: str = "") -> str:
    """Rank video clips using OpenRouter API with Deepseek."""
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
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=1000
    )
    
    return completion.choices[0].message.content

def main():
    start_time = time.time()
    
    api_key = os.getenv("OPEN_ROUTER_KEY")
    if not api_key:
        raise ValueError("Please set the OPEN_ROUTER_KEY environment variable")
    
    site_url = "http://yoursite.com"
    site_name = "Your Site Name"
    json_path = "viral_clips.json"
    
    try:
        load_start = time.time()
        clips = load_clips(json_path)
        load_time = time.time() - load_start
        
        rank_start = time.time()
        result = rank_clips(clips, api_key, site_url, site_name)
        rank_time = time.time() - rank_start
        
        print("\nRanking Results:")
        print("-" * 50)
        print(result)
        
        total_time = time.time() - start_time
        print("\nProcessing Times:")
        print(f"Load Time: {load_time:.2f} seconds")
        print(f"Ranking Time: {rank_time:.2f} seconds")
        print(f"Total Time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()