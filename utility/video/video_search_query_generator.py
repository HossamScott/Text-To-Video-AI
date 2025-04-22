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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")

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


import re

def extract_json_from_text(text):
    """Extract and parse valid segments from malformed JSON-like text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        segments = []
        pattern = re.compile(r'\[\[(\d+\.?\d*),\s*(\d+\.?\d*)\],\s*\[\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\s*\]\]')
        matches = pattern.findall(text)
        for match in matches:
            start, end = float(match[0]), float(match[1])
            keywords = [match[2], match[3], match[4]]
            segments.append([[start, end], keywords])
        return segments if segments else None


def normalize_response(content):
    """Normalize different types of AI responses to standard format."""
    if isinstance(content, dict):
        segments = []
        for k, v in content.items():
            try:
                time_range = json.loads(k.strip())
                if not isinstance(time_range, list) or len(time_range) != 2:
                    continue
                if isinstance(v, list) and len(v) == 3:
                    segments.append([time_range, v])
            except Exception:
                continue
        return segments
    return content



def validate_segment(segment, index):
    """Validate structure and contents of one segment."""
    if not isinstance(segment, list) or len(segment) != 2:
        raise ValueError(f"Segment {index} must be a list with 2 elements")

    time_part, keywords_part = segment

    if not (isinstance(time_part, list) and len(time_part) == 2 and
            all(isinstance(t, (int, float)) for t in time_part)):
        raise ValueError(f"Invalid time range in segment {index}: {time_part}")

    if not (isinstance(keywords_part, list) and len(keywords_part) == 3 and
            all(isinstance(k, str) for k in keywords_part)):
        raise ValueError(f"Invalid keywords in segment {index}: {keywords_part}")


@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=2, backoff_factor=2)
def call_AI_api(script, captions_timed, language="en"):
    try:
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

        print("=== API REQUEST ===")
        print(json.dumps(payload, indent=2))

        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        raw_content = result.get("message", {}).get("content", "").strip()
        clean_content = re.sub(r'```json|```', '', raw_content).strip()

        print("=== API RAW CONTENT ===")
        print(clean_content)

        parsed = extract_json_from_text(clean_content) or {}
        print("=== PARSED JSON OBJECT ===")
        print(json.dumps(parsed, indent=2) if isinstance(parsed, (dict, list)) else parsed)

        normalized = normalize_response(parsed)
        print("=== NORMALIZED RESPONSE ===")
        print(json.dumps(normalized, indent=2) if isinstance(normalized, list) else normalized)

        if not isinstance(normalized, list) or not normalized:
            raise ValueError("No valid segments found in response")

        for i, segment in enumerate(normalized):
            validate_segment(segment, i)

        return normalized

    except json.JSONDecodeError:
        print(f"JSON parsing failed for content:\n{clean_content}")
        raise ValueError("Failed to parse JSON")

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