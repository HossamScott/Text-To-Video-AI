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
        # Test Ollama connection
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        response.raise_for_status()
        return None, OLLAMA_MODEL  # No client object needed for Ollama
    except Exception as e:
        logger.error(f"Ollama connection failed: {str(e)}")
        raise ValueError("Failed to connect to Ollama. Is it running?")

# Initialize model
_, model = get_ai_client()

# Language-specific prompts
PROMPTS = {
    "en": """# Instructions [English]
Given the following video script and timed captions, extract three visually concrete and specific keywords for each time segment that can be used to search for background videos. The keywords should be short and capture the main essence of the sentence. They can be synonyms or related terms. If a caption is vague or general, consider the next timed caption for more context. If a keyword is a single word, try to return a two-word keyword that is visually concrete. If a time frame contains two or more important pieces of information, divide it into shorter time frames with one keyword each. Ensure that the time periods are strictly consecutive and cover the entire length of the video. Each keyword should cover between 2-4 seconds. The output should be in JSON format, like this: [[[t1, t2], ["keyword1", "keyword2", "keyword3"]], [[t2, t3], ["keyword4", "keyword5", "keyword6"]], ...]. Please handle all edge cases, such as overlapping time segments, vague or general captions, and single-word keywords.

Important Guidelines:
- Use only English in your text queries
- Each search string must depict something visual
- The depictions have to be extremely visually concrete
- Return only the JSON response with no extra text""",

    "ar": """# التعليمات [العربية]
بناءً على السيناريو والتعليقات الموقتة التالية، استخرج ثلاث كلمات مفتاحية محددة ومرئية لكل مقطع زمني يمكن استخدامها للبحث عن مقاطع فيديو خلفية. يجب أن تكون الكلمات المفتاحية قصيرة وتلتقط الجوهر الرئيسي للجملة. يمكن أن تكون مرادفات أو مصطلحات ذات صلة. إذا كان التعليق غامضًا أو عامًا، ففكر في التعليق الموقت التالي لمزيد من السياق. إذا كانت الكلمة المفتاحية كلمة واحدة، فحاول إرجاع كلمة مفتاحية مكونة من كلمتين تكون مرئية بشكل ملموس. إذا كان الإطار الزمني يحتوي على جزأين مهمين أو أكثر من المعلومات، فقسمه إلى أطر زمنية أقصر بكلمة مفتاحية واحدة لكل منها. تأكد من أن الفترات الزمنية متتالية بشكل صارم وتغطي الطول الكامل للفيديو. يجب أن تغطي كل كلمة مفتاحية ما بين 2-4 ثوانٍ. يجب أن يكون الإخراج بتنسيق JSON، مثل هذا: [[[t1, t2], ["keyword1", "keyword2", "keyword3"]], [[t2, t3], ["keyword4", "keyword5", "keyword6"]], ...]. يرجى التعامل مع جميع الحالات الطرفية، مثل المقاطع الزمنية المتداخلة، والتعليقات الغامضة أو العامة، والكلمات المفتاحية المكونة من كلمة واحدة.

إرشادات مهمة:
- استخدم فقط اللغة الإنجليزية في استعلامات النص
- يجب أن يصور كل سلسلة بحث شيئًا مرئيًا
- يجب أن تكون التصوير ملموسًا للغاية من الناحية المرئية
- أعد فقط استجابة JSON بدون نص إضافي"""
}

def fix_json(json_str):
    """Clean and fix common JSON formatting issues"""
    json_str = json_str.replace("'", '"')
    json_str = json_str.replace("'", "'").replace("'", "'")
    json_str = json_str.replace(""", '"').replace(""", '"')
    return json_str

def getVideoSearchQueriesTimed(script, captions_timed, language="en"):
    end = captions_timed[-1][0][1]
    try:
        out = [[[0, 0], ""]]
        while out[-1][0][1] != end:
            content = call_AI_api(script, captions_timed, language).replace("'", '"')
            try:
                out = json.loads(content)
            except Exception as e:
                content = fix_json(content.replace("```json", "").replace("```", ""))
                out = json.loads(content)
        return out
    except Exception as e:
        logger.error(f"Error processing response: {str(e)}")
        raise

@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2)
def call_AI_api(script, captions_timed, language="en"):
    """Make API call to Ollama with proper error handling"""
    try:
        # Prepare the input content
        user_content = f"""Script: {script}\nTimed Captions: {json.dumps(captions_timed)}"""
        
        # Prepare the Ollama API request
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": PROMPTS.get(language, PROMPTS["en"])},
                {"role": "user", "content": user_content}
            ],
            "stream": False,
            "format": "json"  # Request JSON response
        }

        # Make the API call
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        # Process the response
        content = response.json()
        text = content["message"]["content"].strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Log the response
        log_response(LOG_TYPE_GPT, script, text)
        
        # Parse the JSON response with error handling
        try:
            if "```json" in text:
                # Extract JSON from markdown
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif isinstance(text, str):
                return json.loads(text)
            return text  # if it's already a dict
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {text}")
            raise ValueError("Invalid JSON response from AI API")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {str(e)}")
        raise ValueError("Failed to communicate with Ollama API")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
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