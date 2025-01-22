from openai import OpenAI
import json
import os
from typing import List, Dict

def load_clips(json_path: str) -> List[Dict]:
    """
    Load clips from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing clips data
    
    Returns:
        List of clip dictionaries
    """
    try:
        with open(json_path, 'r') as file:
            clips = json.load(file)
        return clips
    except FileNotFoundError:
        raise FileNotFoundError(f"Clips file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_path}")

def rank_clips(clips: List[Dict], api_key: str, site_url: str = "", site_name: str = "") -> str:
    """
    Rank video clips using OpenRouter API with GPT-4 Turbo.
    
    Args:
        clips: List of clip dictionaries
        api_key: OpenRouter API key
        site_url: Optional site URL for OpenRouter rankings
        site_name: Optional site name for OpenRouter rankings
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Construct the prompt
    prompt = f"""Please rank these clips based on their potential virality:
{json.dumps(clips, indent=2)}

For each clip provided (and ONLY these clips), give:
1. Clip name and length
2. A brief explanation of virality potential based on length and content type

Keep explanations focused and concise."""

    # Create the API request
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        },
        model="openai/gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that ranks video clips. You must ONLY rank the clips provided in the input, with no additional clips. Keep explanations brief and focused on virality potential."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return completion.choices[0].message.content

def main():
    # Get API key from environment variable
    api_key = os.getenv("OPEN_ROUTER_KEY")
    if not api_key:
        raise ValueError("Please set the OPEN_ROUTER_KEY environment variable")
    
    # Optional site information
    site_url = "http://yoursite.com"  # Replace with your site URL
    site_name = "Your Site Name"      # Replace with your site name
    
    # Path to your JSON file
    json_path = "v.json"  # You can also make this a command line argument
    
    try:
        # Load clips from JSON file
        clips = load_clips(json_path)
        
        # Get rankings
        result = rank_clips(clips, api_key, site_url, site_name)
        print("\nRanking Results:")
        print("-" * 50)
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()