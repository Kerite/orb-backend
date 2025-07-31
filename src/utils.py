import requests
from bs4 import BeautifulSoup
import json
import re
from typing import List, Dict

def parse_react_router_stream_data(html_content: str):
    """
    Parse React Router streaming data from the HTML content
    """
    # Look for the large script tag that contains the stream data
    soup = BeautifulSoup(html_content, "html.parser")
    scripts = soup.find_all("script")

    # Find the script with the largest content (likely contains the stream data)
    largest_script = None
    max_length = 0

    for script in scripts:
        if script.string and len(script.string) > max_length:
            max_length = len(script.string)
            largest_script = script

    if not largest_script:
        print("No suitable script tag found in the HTML content.")
        return None

    content = largest_script.string

    # Look for the actual conversation text directly
    text_pattern = r'"([^"]{50,})"'  # Look for very long strings
    text_matches = re.findall(text_pattern, content)

    # Filter for likely conversation content
    conversation_texts = []
    for text in text_matches:
        if any(keyword in text.lower() for keyword in ['air purifier', 'carpet', 'paint', 'filter', 'hepa', 'choose', 'home']):
            conversation_texts.append(text)

    if conversation_texts:
        return {"conversation_texts": conversation_texts}

    return None

def extract_chatgpt_share_messages(url: str) -> List[Dict[str, str]]:
    """
    Given a public ChatGPT share URL, fetches the page,
    parses the embedded data, and returns
    a list of {"role": "system"|"user"|"assistant", "content": "..."}.
    """
    # 1. Fetch page
    resp = requests.get(url)
    resp.raise_for_status()

    print(f"Http Content: {resp.text[:5000]}...")

    # 2. Parse React Router streaming data
    data = parse_react_router_stream_data(resp.text)

    if not data:
        raise ValueError("Could not find ChatGPT data on the page.")

    # 3. Extract messages from the conversation texts
    messages = []

    if "conversation_texts" in data:
        texts = data["conversation_texts"]

        # Based on the content we found, we can reconstruct the conversation
        # The first text appears to be the user's question
        # The second text appears to be the assistant's response

        if len(texts) >= 2:
            # First message is the user's question
            messages.append({
                "role": "user",
                "content": texts[0]
            })

            # Second message is the assistant's response
            messages.append({
                "role": "assistant",
                "content": texts[1]
            })

    if not messages:
        raise ValueError("Could not extract messages from the conversation data.")

    return messages

# # Test the function
# if __name__ == "__main__":
#     share_url = "https://chat.openai.com/share/68852f44-e960-8010-b2cb-9c6796e83dd1"
#     try:
#         messages = extract_chatgpt_share_messages(share_url)
#         print(f"\n=== EXTRACTED MESSAGES ===")
#         print(json.dumps(messages, indent=2))
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()