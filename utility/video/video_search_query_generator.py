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
    "en": """You are a video search query generator. Generate exactly 3 visual keywords per time segment.

STRICT RESPONSE FORMAT (JSON ONLY):
[
  [[start1, end1], ["kw1", "kw2", "kw3"]],
  [[start2, end2], ["kw4", "kw5", "kw6"]]
]

RULES:
1. Time segments must cover the full duration consecutively
2. Keywords must be visual and concrete
3. Only return the JSON array, no other text

BAD EXAMPLE: {"result": [...]}
GOOD EXAMPLE: [[[0,2],["city","buildings"]], [[2,4],["park","trees"]]]

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


def extract_segments(text):
    """Multi-stage extraction with AI fallback"""
    # Attempt 1: Direct JSON parsing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except:
        pass

    # Attempt 2: Find JSON-like patterns
    json_pattern = r'\[(\[\[.*?\]\].*?\])\]'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(f"[{matches[0]}]")
        except:
            pass

    # Attempt 3: Line-by-line pattern matching
    segments = []
    pattern = re.compile(
        r'\[?\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]\s*,\s*\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]\s*\]?'
    )
    for match in pattern.finditer(text):
        try:
            segments.append([
                [float(match.group(1)), float(match.group(2))],
                [match.group(3), match.group(4), match.group(5)]
            ])
        except:
            continue

    # Final Attempt: Ask AI to fix formatting
    if not segments:
        return self_correct_format(text)
    
    return segments

def self_correct_format(malformed_text):
    """Use AI to fix malformed responses"""
    correction_prompt = f"""Fix this malformed JSON into the correct format:
    
    MALFORMED INPUT:
    {malformed_text}
    
    CORRECT FORMAT:
    [[[start,end], ["kw1","kw2","kw3"]], ...]
    
    OUTPUT ONLY THE CORRECTED JSON:"""
    
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": correction_prompt}],
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
        return json.loads(response.json()["message"]["content"])
    except Exception as e:
        logger.error(f"Format correction failed: {str(e)}")
        return []

def normalize_segments(raw_data):
    """Normalize various response formats"""
    # Handle string input
    if isinstance(raw_data, str):
        return extract_segments(raw_data)
    
    # Handle dictionary responses
    if isinstance(raw_data, dict):
        for key in ["segments", "results", "data"]:
            if key in raw_data:
                return raw_data[key]
        return [v for k,v in raw_data.items() if isinstance(v, list)]
    
    # Handle list variants
    return [
        [item[:2], item[2:5]] if len(item) > 2 else item
        for item in raw_data 
        if isinstance(item, (list, tuple))
    ]

def validate_segment(segment, index):
    """Flexible validation with auto-correction"""
    if not isinstance(segment, list) or len(segment) < 2:
        raise ValueError(f"Invalid segment structure at index {index}")
    
    # Normalize time range
    time_part = segment[0]
    if not isinstance(time_part, (list, tuple)) or len(time_part) != 2:
        time_part = [float(time_part[0]), float(time_part[1])] if isinstance(time_part, list) else [0, 0]
    
    # Normalize keywords
    keywords = segment[1] if len(segment) > 1 else []
    if isinstance(keywords, str):
        keywords = [kw.strip() for kw in keywords.split(",")[:3]]
    elif isinstance(keywords, (list, tuple)):
        keywords = [str(kw) for kw in keywords[:3]]
    
    return [time_part, keywords]


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

    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
    response.raise_for_status()
    result = response.json()

        # Process response
    raw_content = result.get("message", {}).get("content", "")
    logger.debug(f"Raw AI Response:\n{raw_content}")
    
    # Multi-stage processing
    parsed = extract_segments(raw_content)
    logger.debug(f"Initial Parsing:\n{json.dumps(parsed, indent=2)}")
    
    normalized = normalize_segments(parsed or raw_content)
    logger.debug(f"Normalized Segments:\n{json.dumps(normalized, indent=2)}")
    
    # Validate and correct each segment
    final_segments = []
    for i, segment in enumerate(normalized):
        try:
            final_segments.append(validate_segment(segment, i))
        except Exception as e:
            logger.warning(f"Skipping invalid segment {i}: {str(e)}")
    
    if not final_segments:
        raise ValueError("No valid segments after validation")
    
    # Ensure temporal continuity
    return ensure_temporal_continuity(final_segments, captions_timed[-1][0][1])

def ensure_temporal_continuity(segments, total_duration):
    """Fill gaps and ensure full coverage"""
    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x[0][0])
    
    # Fill time gaps
    last_end = 0
    for seg in sorted_segments:
        if seg[0][0] > last_end:
            sorted_segments.append([[last_end, seg[0][0]], ["general", "scene", "background"]])
        last_end = max(last_end, seg[0][1])
    
    # Extend last segment if needed
    if last_end < total_duration:
        sorted_segments.append([[last_end, total_duration], ["ending", "scene", "background"]])
    
    return sorted(sorted_segments, key=lambda x: x[0][0])


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