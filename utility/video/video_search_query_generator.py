import os
import json
import re
import requests
import logging
from utility.retry_utils import retry_api_call, handle_common_errors

logger = logging.getLogger(__name__)

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
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


def extract_segments(text: str):
    """Try multiple strategies to parse JSON-like segment output."""
    # 1) direct JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # 2) grab the first [...]] block
    json_pattern = r'\[(\[\[.*?\]\].*?\])\]'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(f"[{matches[0]}]")
        except Exception:
            pass

    # 3) line-by-line regex
    segments = []
    pattern = re.compile(
        r'\[?\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]\s*,\s*\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]\s*\]?'
    )
    for m in pattern.finditer(text):
        try:
            segments.append([
                [float(m.group(1)), float(m.group(2))],
                [m.group(3), m.group(4), m.group(5)]
            ])
        except Exception:
            continue

    # 4) if still empty, ask the model to fix it
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
        "model":  OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
        "format":  "json"
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
    """Turn dicts or mixed lists into a flat list of [ [start,end], [kw1,kw2,kw3] ]."""
    if isinstance(raw, str):
        return extract_segments(raw)
    if isinstance(raw, dict):
        for key in ("query_segments", "segments", "data", "results"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        # fallback: any list value
        return [v for v in raw.values() if isinstance(v, list)]
    if isinstance(raw, (list, tuple)):
        normalized = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                time_part = item[0]
                kw_part   = item[1]
                normalized.append([time_part, kw_part])
        return normalized
    return []


def validate_segment(seg, idx):
    """Ensure each segment is two floats + 3 keywords."""
    if not (isinstance(seg, list) and len(seg) == 2):
        raise ValueError(f"Segment {idx} not [time,kw]")
    t, kws = seg
    if not (isinstance(t, (list, tuple)) and len(t) == 2):
        raise ValueError(f"Segment {idx} bad time: {t}")
    if not (isinstance(kws, (list, tuple)) and len(kws) == 3):
        raise ValueError(f"Segment {idx} needs 3 keywords: {kws}")
    return [[float(t[0]), float(t[1])], [str(kws[0]), str(kws[1]), str(kws[2])]]


def preprocess_captions(captions):
    """Remove invalid entries, accept lists or tuples, normalize to lists."""
    cleaned = []
    for cap in captions:
        if not isinstance(cap, (list, tuple)) or len(cap) != 2:
            logger.warning(f"Skipping invalid caption (not list/tuple of len2): {cap}")
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
    """Group captions into chunks of ~max_seg seconds if initial call fails."""
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
    """Core LLM call with full debug prints."""
    sys_prompt = PROMPTS.get(language, PROMPTS["en"])
    user_payload = f"Script: {script}\nTimed Captions: {json.dumps(captions)}"
    payload = {
        "model":   OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role":   "user", "content": user_payload}
        ],
        "stream": False,
        "format": "json"
    }

    print("=== API REQUEST ===")
    print(json.dumps(payload, indent=2))

    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("message", {}).get("content", "")
    logger.debug(f"Raw AI Response:\n{raw}")

    # parse → normalize → validate
    parsed     = extract_segments(raw)
    logger.debug(f"Parsed segments:\n{json.dumps(parsed, indent=2)}")
    normalized = normalize_segments(parsed)
    logger.debug(f"Normalized segments:\n{json.dumps(normalized, indent=2)}")

    final = []
    for i, seg in enumerate(normalized):
        try:
            final.append(validate_segment(seg, i))
        except Exception as e:
            logger.warning(f"Skipping invalid segment {i}: {e}")

    if not final:
        raise ValueError("No valid segments after validation")

    return ensure_temporal_continuity(final, captions[-1][0][1])


def ensure_temporal_continuity(segs, total_dur):
    """Fill gaps so you cover [0, total_dur] continuously."""
    segs = sorted(segs, key=lambda x: x[0][0])
    last_end = 0.0
    filled = []
    for s in segs:
        if s[0][0] > last_end:
            filled.append([[last_end, s[0][0]], ["general", "scene", "background"]])
        filled.append(s)
        last_end = max(last_end, s[0][1])
    if last_end < total_dur:
        filled.append([[last_end, total_dur], ["ending", "scene", "background"]])
    return sorted(filled, key=lambda x: x[0][0])


def getVideoSearchQueriesTimed(script, captions, language="en"):
    captions = preprocess_captions(captions)
    if not captions:
        raise ValueError("Empty or invalid captions data")

    try:
        segments = call_AI_api(script, captions, language)
    except Exception as e:
        logger.warning(f"Primary AI call failed: {e}. Trying chunked captions...")
        segments = []
        for chunk in chunk_captions(captions):
            try:
                segments += call_AI_api(script, chunk, language)
            except Exception as sub_e:
                logger.error(f"Chunked call failed: {sub_e}")

    end_time = captions[-1][0][1]
    if segments and segments[-1][0][1] < end_time:
        logger.warning(f"Coverage gap: last segment ends {segments[-1][0][1]}s, video ends {end_time}s")

    return ensure_temporal_continuity(segments, end_time)


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
