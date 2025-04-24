import os
import json
import re
import requests
import logging
from utility.retry_utils import retry_api_call, handle_common_errors

logger = logging.getLogger(__name__)

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")

PROMPTS = {
    "en": """You are a video search query generator. Generate exactly 3 visual keywords per time segment.
STRICT RESPONSE FORMAT (JSON ONLY):
[
  [[start1, end1], ["kw1", "kw2", "kw3"]],
  [[start2, end2], ["kw4", "kw5", "kw6"]]
]
CRITICAL RULES:
1. Times must be NUMBERS (e.g., 0.0 not "0.0")
2. EXACT structure: [[[number,number],["word","word","word"]]
3. No duplicate keys or nested objects
4. Keywords must be English visual concepts

BAD EXAMPLES TO AVOID:
{"segments": [...]}  # Object wrapper
[["0.0", "1.0"], ["kw1","kw2","kw3"]]  # String times
[[[0,2],["city"]], ...]  # Only 1 keyword

GOOD EXAMPLE:
[[[0,2],["city","buildings","skyline"]], [[2,4],["park","trees","path"]]]

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

def extract_segments(text: str):
    """More robust parsing with numeric conversion"""
    # FIX: Improved regex for malformed JSON
    json_pattern = r'(?s)\[(?:\[\[[^]]+\]\])(?:,\s*\[\[[^]]+\]\])*\]'
    
    # Attempt direct JSON parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    
    # FIX: New strategy - find all [[time,time], [kw,kw,kw]] patterns
    segments = []
    pattern = re.compile(
        r'\[?\s*\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]\s*,\s*\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]\s*\]?',
        re.DOTALL
    )
    for match in pattern.finditer(text):
        try:
            start = float(match.group(1))
            end = float(match.group(2))
            kws = [match.group(3), match.group(4), match.group(5)]
            segments.append([[start, end], kws])
        except Exception:
            continue
    
    # FIX: Fallback to self-correction only if no segments found
    if not segments:
        return self_correct_format(text)
    
    return segments


def self_correct_format(malformed: str):
    """Use the LLM to re-format broken JSON into the strict array-of-segments format."""
    prompt = f"""Fix this malformed JSON into the exact format:
[[[start,end], ["kw1","kw2","kw3"]], ...]

MALFORMED INPUT:
{malformed}

OUTPUT ONLY THE CORRECTED JSON ARRAY."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json"
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return json.loads(content)
    except Exception as e:
        logger.error(f"Format correction failed: {e}")
        return []


def normalize_segments(raw):
    """Handle more varied structures"""
    # FIX: Added support for {'segments': [...]} and similar
    if isinstance(raw, dict):
        candidates = ['segments', 'results', 'data']
        for key in candidates:
            if isinstance(raw.get(key), list):
                return raw[key]
        return [v for v in raw.values() if isinstance(v, list)]
    
    # Handle list of mixed types
    normalized = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append([item[0], item[1]])
        elif isinstance(item, dict):
            normalized.append([item.get('time'), item.get('keywords')])
    
    return normalized


def validate_segment(seg, idx):
    """More lenient validation with type conversion"""
    # FIX: Allow flexible structure as long as we can extract times/keywords
    try:
        # Handle different structures
        if isinstance(seg, dict):  # Convert {"start":0, "end":1, "keywords":[...]} 
            t = [seg.get('start', 0), seg.get('end', 0)]
            kws = seg.get('keywords', [])
        else:
            t = seg[0] if len(seg) > 0 else [0, 0]
            kws = seg[1] if len(seg) > 1 else []
        
        # Convert to floats and strings
        start = float(t[0]) if len(t) > 0 else 0.0
        end = float(t[1]) if len(t) > 1 else 0.0
        kws = [str(k).strip() for k in kws][:3]  # Take first 3
        
        # Pad keywords if less than 3
        while len(kws) < 3:
            kws.append("general")
            
        return [[start, end], kws]
    except Exception as e:
        logger.warning(f"Segment {idx} partial validation: {e}")
        return [[0.0, 0.0], ["general", "scene", "background"]]


def preprocess_captions(captions):
    """Normalize captions to [[start,end], text]."""
    cleaned = []
    for cap in captions:
        if not isinstance(cap, (list, tuple)) or len(cap) != 2:
            logger.warning(f"Skipping invalid caption (not list/tuple len2): {cap}")
            continue
        time, text = cap
        if not isinstance(time, (list, tuple)) or len(time) != 2:
            logger.warning(f"Skipping caption with bad timing: {cap}")
            continue
        start, end = time
        if not all(isinstance(x, (int, float)) for x in (start, end)):
            logger.warning(f"Skipping caption with non-numeric time: {cap}")
            continue
        if not isinstance(text, str) or not text.strip():
            logger.warning(f"Skipping caption with empty text: {cap}")
            continue
        cleaned.append([[float(start), float(end)], text.strip()])
    return cleaned


def chunk_captions(captions, max_seg=4.0):
    """Group captions into ~max_seg-second chunks if initial call fails."""
    if not captions:
        return []
    chunks, cur, seg_start = [], [], captions[0][0][0]
    for cap in captions:
        cur.append(cap)
        if cap[0][1] - seg_start >= max_seg:
            chunks.append(cur)
            cur = []
            seg_start = cap[0][1]
    if cur:
        chunks.append(cur)
    return chunks


@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=2, backoff_factor=2)
def call_AI_api(script, captions, language="en"):
    """Call the model, parse, normalize, validate, and fallback if needed."""
    sys_prompt = PROMPTS.get(language, PROMPTS["en"])
    user_payload = f"Script: {script}\nTimed Captions: {json.dumps(captions)}"
    payload = {
        "model":   OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role":   "user", "content": user_payload}
        ],
        "stream":  False,
        "format": "json"
    }

    # 1) Show the request
    print("=== API REQUEST ===")
    print(json.dumps(payload, indent=2))

    # 2) Call Ollama
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    # 3) Print the raw AI response
    raw = data.get("message", {}).get("content", "")
    print("=== API RAW CONTENT ===")
    print(raw)

    # 4) Parse → normalize
    parsed     = extract_segments(raw)
    print("=== PARSED ===")
    print(parsed)

    normalized = normalize_segments(parsed)
    print("=== NORMALIZED ===")
    print(normalized)

    # 5) Validate segments strictly
    final = []
    for i, seg in enumerate(normalized):
        try:
            final.append(validate_segment(seg, i))
        except Exception as e:
            logger.warning(f"Skipping invalid segment {i}: {e}")

    # 6) Fallback: if none passed validation, use the normalized list as-is
    if not final:
        if normalized:
            logger.warning("No valid segments after validation—falling back to normalized segments")
            final = normalized
        else:
            raise ValueError("No valid segments after validation")

    # 7) Ensure full coverage
    total_dur = captions[-1][0][1]
    return ensure_temporal_continuity(final, total_dur)


def ensure_temporal_continuity(segs, total_dur):
    """Type-safe continuity check"""
    # FIX: Convert all times to floats first
    converted = []
    for seg in segs:
        try:
            start = float(seg[0][0])
            end = float(seg[0][1])
            converted.append([[start, end], seg[1]])
        except Exception:
            continue
    
    # Sort by start time after conversion
    sorted_segs = sorted(converted, key=lambda x: x[0][0])
    
    # Fill gaps with type-safe comparisons
    last_end = 0.0
    filled = []
    for s in sorted_segs:
        start = float(s[0][0])
        end = float(s[0][1])
        
        if start > last_end:
            filled.append([[last_end, start], ["general", "scene", "background"]])
        
        filled.append(s)
        last_end = max(last_end, end)
    
    # Handle final gap
    if last_end < float(total_dur):
        filled.append([[last_end, float(total_dur)], ["ending", "scene", "background"]])
    
    return filled

def getVideoSearchQueriesTimed(script, captions, language="en"):
    """Preprocess captions → call the API → return final segments."""
    caps = preprocess_captions(captions)
    if not caps:
        raise ValueError("Empty or invalid captions data")

    try:
        return call_AI_api(script, caps, language=language)
    except Exception as e:
        logger.warning(f"Primary call failed: {e}. Retrying with caption chunks...")
        merged = []
        for chunk in chunk_captions(caps):
            try:
                merged.extend(call_AI_api(script, chunk, language=language))
            except Exception as sub_e:
                logger.error(f"Chunk retry failed: {sub_e}")
        if not merged:
            raise
        return merged


def merge_empty_intervals(segments):
    merged, i = [], 0
    while i < len(segments):
        interval, url = segments[i]
        if url is None:
            j = i + 1
            while j < len(segments) and segments[j][1] is None:
                j += 1
            if merged and merged[-1][1] is not None and merged[-1][0][1] == interval[0]:
                merged[-1] = [[merged[-1][0][0], segments[j-1][0][1]], merged[-1][1]]
            else:
                merged.append([interval, None])
            i = j
        else:
            merged.append([interval, url])
            i += 1
    return merged