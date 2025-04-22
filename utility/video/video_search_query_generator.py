import os
import json
import re
import requests
from datetime import datetime
from utility.utils import log_response, LOG_TYPE_GPT
from utility.retry_utils import retry_api_call, handle_common_errors
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

def get_ai_client():
    """Initialize and return Ollama configuration"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        response.raise_for_status()
        return None, OLLAMA_MODEL
    except Exception as e:
        logger.error(f"Ollama connection failed: {str(e)}")
        raise ValueError("Failed to connect to Ollama. Is it running?")

# Initialize model
_, model = get_ai_client()

# Language-specific prompts
PROMPTS = {
    "en": """You are a video search query generator. For each time segment in the captions, generate exactly 3 visually concrete keywords for finding background videos.

STRICT OUTPUT FORMAT (JSON ONLY):
[
  [[start1, end1], ["keyword1", "keyword2", "keyword3"]],
  [[start2, end2], ["keyword4", "keyword5", "keyword6"]]
]

RULES:
1. Keywords MUST be:
   - Visually concrete (e.g., "futuristic city" not "future")
   - In English only
   - 1-3 words each
   - Specific to the time segment
2. Time segments MUST:
   - Be consecutive
   - Cover the entire video duration
   - Each be 2-4 seconds
3. OUTPUT MUST:
   - Be pure JSON (no markdown, no explanations)
   - Contain exactly 3 keywords per segment
   - Maintain the exact specified format

INPUT:
Script: {script}
Timed Captions: {captions}

OUTPUT (JSON ONLY):""",

    "ar": """أنت مولد استعلامات بحث الفيديو. لكل مقطع زمني في التعليقات، قم بإنشاء 3 كلمات مفتاحية مرئية ملموسة للعثور على مقاطع الفيديو الخلفية.

تنسيق الإخراج الصارم (JSON فقط):
[
  [[البداية1, النهاية1], ["الكلمة1", "الكلمة2", "الكلمة3"]],
  [[البداية2, النهاية2], ["الكلمة4", "الكلمة5", "الكلمة6"]]
]

القواعد:
1. يجب أن تكون الكلمات المفتاحية:
   - مرئية ملموسة (مثال: "مدينة مستقبلية" وليس "مستقبل")
   - باللغة الإنجليزية فقط
   - من 1-3 كلمات لكل منها
   - محددة للمقطع الزمني
2. المقاطع الزمنية يجب أن:
   - تكون متتالية
   - تغطي مدة الفيديو كاملة
   - كل منها من 2-4 ثواني
3. يجب أن يكون الإخراج:
   - JSON خالص (بدون تنسيق markdown، بدون شروحات)
   - يحتوي على 3 كلمات مفتاحية لكل مقطع بالضبط
   - يحافظ على التنسيق المحدد بدقة

المدخلات:
النص: {script}
التعليقات الموقتة: {captions}

الإخراج (JSON فقط):"""
}


def normalize_response(content):
    """Normalize different response formats to standard structure"""
    if isinstance(content, dict):
        if "segments" in content:
            return content["segments"]
        if "results" in content:
            return content["results"]
        return [content]
    return content

def extract_json_from_text(text):
    """Extract JSON structures from malformed text responses"""
    try:
        # First try direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback to pattern matching
        json_match = re.search(r'(\[\[.*?\]\])', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        return None

# def fix_json(json_str):
#     """Clean and fix common JSON formatting issues"""
#     json_str = json_str.replace("'", '"')
#     json_str = json_str.replace("'", "'").replace("'", "'")
#     json_str = json_str.replace(""", '"').replace(""", '"')
#     return json_str

# def getVideoSearchQueriesTimed(script, captions_timed, language="en"):
#     if not captions_timed:
#         raise ValueError("Empty captions data")
    
#     try:
#         # Get the raw response
#         response = call_AI_api(script, captions_timed, language)
        
#         # Validate the response structure
#         if not isinstance(response, list):
#             raise ValueError(f"Expected list, got {type(response)}")
            
#         if not all(
#             isinstance(segment, list) and 
#             len(segment) == 2 and
#             isinstance(segment[0], list) and
#             isinstance(segment[1], list)
#             for segment in response
#         ):
#             raise ValueError("Malformed response segments")
            
#         return response
        
#     except Exception as e:
#         logger.error(f"Error processing response: {str(e)}")
#         raise
@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=2, backoff_factor=2)
def call_AI_api(script, captions_timed, language="en"):
    try:
        # Prepare request payload
        system_prompt = PROMPTS.get(language, PROMPTS["en"])
        user_content = f"Script: {script}\nTimed Captions: {json.dumps(captions_timed)}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "stream": False,
            "format": "json"
        }

        # Log request details
        logger.debug("\n=== API REQUEST ===")
        logger.debug(f"Model: {model}")
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug(f"User Content:\n{user_content}")
        logger.debug("===================")

        # Make API call
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        # Process response
        result = response.json()
        raw_content = result.get("message", {}).get("content", "")
        
        # Log full response
        logger.debug("\n=== API RESPONSE ===")
        logger.debug("Full Response:")
        logger.debug(json.dumps(result, indent=2))
        logger.debug(f"Raw Content:\n{raw_content}")
        logger.debug("====================")

        # Clean and parse content
        clean_content = re.sub(r'```json|```', '', raw_content).strip()
        logger.debug("\n=== PROCESSED CONTENT ===")
        logger.debug(clean_content)
        logger.debug("=======================")

        # Extract JSON from response
        parsed = extract_json_from_text(clean_content) or {}
        logger.debug("\n=== PARSED STRUCTURE ===")
        logger.debug(f"Type: {type(parsed)}")
        logger.debug(f"Content: {parsed}")
        logger.debug("======================")

        # Normalize and validate response
        normalized = normalize_response(parsed)
        
        if not isinstance(normalized, list):
            logger.error(f"Invalid response type: {type(normalized)}")
            raise ValueError(f"Expected list, got {type(normalized)}")

        # Validate segments - FIXED SYNTAX HERE
        for i, segment in enumerate(normalized):
            if not (isinstance(segment, list) and len(segment) == 2):
                logger.error(f"Invalid segment {i}: {segment}")
                raise ValueError(f"Segment {i} has invalid format")
            if not (isinstance(segment[0], list) and isinstance(segment[1], list)):
                logger.error(f"Invalid segment values {i}: {segment}")
                raise ValueError(f"Segment {i} has invalid values")

        return normalized

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode failed. Content: {clean_content}")
        raise ValueError(f"Failed to parse JSON: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed. Status: {response.status_code if 'response' in locals() else 'N/A'}")
        raise ValueError(f"API communication error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
        if 'result' in locals():
            logger.error(f"Last response: {json.dumps(result, indent=2)}")
        raise ValueError(f"API processing error: {str(e)}")

# @handle_common_errors
# @retry_api_call(max_retries=3, initial_delay=2, backoff_factor=2)
# def call_AI_api(script, captions_timed, language="en"):
#     try:
#         payload = {
#             "model": model,
#             "messages": [
#                 {"role": "system", "content": PROMPTS.get(language, PROMPTS["en"])},
#                 {"role": "user", "content": f"Script: {script}\nTimed Captions: {json.dumps(captions_timed)}"}
#             ],
#             "stream": False,
#             "format": "json"
#         }

#         response = requests.post(
#             f"{OLLAMA_HOST}/api/chat",
#             json=payload,
#             timeout=120
#         )
#         response.raise_for_status()

#         # Extract and clean the response
#         result = response.json()
#         content = result["message"]["content"].strip()
        
#         # Remove any markdown formatting
#         if content.startswith("```json"):
#             content = content[7:-3].strip()
#         elif content.startswith("```"):
#             content = content[3:-3].strip()
            
#         # Parse the JSON
#         try:
#             parsed = json.loads(content)
#             if not isinstance(parsed, list):
#                 raise ValueError("Response is not a list")
#             return parsed
#         except json.JSONDecodeError:
#             # Try to extract JSON from malformed response
#             json_match = re.search(r'\[\[\[.*\]\]\]', content)
#             if json_match:
#                 return json.loads(json_match.group(0))
#             raise

#     except Exception as e:
#         logger.error(f"API call failed: {str(e)}")
#         raise ValueError(f"API processing error: {str(e)}")
def getVideoSearchQueriesTimed(script, captions_timed, language="en"):
    if not captions_timed:
        raise ValueError("Empty captions data")
    
    try:
        response = call_AI_api(script, captions_timed, language)
        end_time = captions_timed[-1][0][1]
        
        # Verify coverage of video duration
        last_segment_end = response[-1][0][1] if response else 0
        if last_segment_end < end_time:
            logger.warning(f"Missing coverage: {last_segment_end}/{end_time}")
            
        return response
        
    except Exception as e:
        logger.error(f"""
        Query generation failed. Possible fixes:
        1. Check model compatibility (using {model})
        2. Verify prompt adherence
        3. Test with simpler input
        Error: {str(e)}
        """)
        raise

def merge_empty_intervals(segments):
    """Merge consecutive empty intervals in the video segments"""
    merged = []
    i = 0
    while i < len(segments):
        interval, url = segments[i]
        if url is None:
            j = i + 1
            while j < len(segments) and segments[j][1] is None:
                j += 1
            
            if i > 0:
                prev_interval, prev_url = merged[-1]
                if prev_url is not None and prev_interval[1] == interval[0]:
                    merged[-1] = [[prev_interval[0], segments[j-1][0][1]], prev_url]
                else:
                    merged.append([interval, prev_url])
            else:
                merged.append([interval, None])
            i = j
        else:
            merged.append([interval, url])
            i += 1
    return merged
# def merge_empty_intervals(segments):
#     """Merge consecutive empty intervals in the video segments"""
#     merged = []
#     i = 0
#     while i < len(segments):
#         interval, url = segments[i]
#         if url is None:
#             j = i + 1
#             while j < len(segments) and segments[j][1] is None:
#                 j += 1
            
#             if i > 0:
#                 prev_interval, prev_url = merged[-1]
#                 if prev_url is not None and prev_interval[1] == interval[0]:
#                     merged[-1] = [[prev_interval[0], segments[j-1][0][1]], prev_url]
#                 else:
#                     merged.append([interval, prev_url])
#             else:
#                 merged.append([interval, None])
#             i = j
#         else:
#             merged.append([interval, url])
#             i += 1
#     return merged