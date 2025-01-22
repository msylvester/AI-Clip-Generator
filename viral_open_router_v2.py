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

# Rest of the code remains the same...