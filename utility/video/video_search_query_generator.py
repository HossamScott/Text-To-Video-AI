import os
import json
import re
import requests
import logging
from utility.retry_utils import retry_api_call, handle_common_errors

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

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


def extract_json_from_text(text):
    """Extract valid [[start, end], [kw1, kw2, kw3]] segments from possibly malformed JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        segments = []
        pattern = re.compile(
            r'\[\[\s*(\d+\.?\d*),\s*(\d+\.?\d*)\s*\],\s*\[\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\s*\]\]'
        )
        for match in pattern.finditer(text):
            start, end = float(match.group(1)), float(match.group(2))
            keywords = [match.group(3), match.group(4), match.group(5)]
            segments.append([[start, end], keywords])
        return segments if segments else None


def normalize_response(content):
    """Normalize different AI return structures into a list of segments."""
    if isinstance(content, dict):
        # Primary: look for "query_segments"
        if "query_segments" in content and isinstance(content["query_segments"], list):
            return content["query_segments"]
        # Fallback: dict with JSON-string keys
        segments = []
        for k, v in content.items():
            try:
                time_range = json.loads(k)
                if isinstance(time_range, list) and len(time_range) == 2 \
                   and isinstance(v, list) and len(v) == 3:
                    segments.append([time_range, v])
            except Exception:
                continue
        return segments
    return content


def validate_segment(segment, index):
    """Ensure each segment is [ [start, end], [kw1,kw2,kw3] ] with correct types."""
    if not (isinstance(segment, list) and len(segment) == 2):
        raise ValueError(f"Segment {index} must be a list of two elements")
    time_part, keywords = segment
    if not (
        isinstance(time_part, list)
        and len(time_part) == 2
        and all(isinstance(t, (int, float)) for t in time_part)
    ):
        raise ValueError(f"Segment {index} has invalid time range: {time_part}")
    if not (
        isinstance(keywords, list)
        and len(keywords) == 3
        and all(isinstance(k, str) for k in keywords)
    ):
        raise ValueError(f"Segment {index} has invalid keywords: {keywords}")


@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=2, backoff_factor=2)
def call_AI_api(script, captions_timed, language="en"):
    """Call Ollama to generate video search keywords, with full debug prints."""
    system_prompt = PROMPTS.get(language, PROMPTS["en"])
    user_content = f"Script: {script}\nTimed Captions: {json.dumps(captions_timed)}"
    payload = {
        "model": OLLAMA_MODEL,
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

    raw_content = result.get("message", {}).get("content", "")
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


def getVideoSearchQueriesTimed(script, captions_timed, language="en"):
    """
    Wrapper to call call_AI_api and ensure coverage of the entire video duration.
    """
    if not captions_timed:
        raise ValueError("Empty captions data")

    segments = call_AI_api(script, captions_timed, language)
    end_time = captions_timed[-1][0][1]
    last_end = segments[-1][0][1] if segments else 0
    if last_end < end_time:
        logger.warning(f"Missing coverage: segments end at {last_end}s / video ends at {end_time}s")

    return segments


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