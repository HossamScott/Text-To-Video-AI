import os
from openai import OpenAI
from groq import Groq
import json
import re
from datetime import datetime
from utility.utils import log_response, LOG_TYPE_GPT
from utility.retry_utils import retry_api_call, handle_common_errors
import logging

# Configure logging
logger = logging.getLogger(__name__)

def get_ai_client():
    """Initialize and return the appropriate AI client with fallback support"""
    # Try OpenRouter first
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "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")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    if openrouter_key:
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )
            # Test the connection
            client.models.list()
            return client, "deepseek/deepseek-chat-v3-0324:free"
        except Exception as e:
            logger.warning(f"OpenRouter connection failed: {str(e)}")

    # openai_key = os.environ.get("OPENAI_API_KEY")
    # if openai_key:
    #     try:
    #         client = OpenAI(api_key=openai_key)
    #         return client, "gpt-4o"
    #     except Exception as e:
    #         logger.warning(f"OpenAI connection failed: {str(e)}")

    # groq_key = os.environ.get("GROQ_API_KEY")
    # if groq_key:
    #     try:
    #         client = Groq(api_key=groq_key)
    #         return client, "mixtral-8x7b-32768"
    #     except Exception as e:
    #         logger.warning(f"Groq connection failed: {str(e)}")

    raise ValueError("No valid API provider configuration found. Please set environment variables.")

# Initialize client and model
try:
    client, model = get_ai_client()
except Exception as e:
    logger.error(f"Failed to initialize AI client: {str(e)}")
    raise

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
    json_str = json_str.replace("'", '"')
    json_str = json_str.replace("’", "'").replace("‘", "'")
    json_str = json_str.replace("“", '"').replace("”", '"')
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
        print("Error processing response:", e)
    return None
@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2)
def call_AI_api(script, captions_timed, language="en"):
    """Make API call to OpenRouter with proper headers and error handling"""
    try:
        # Prepare the input content
        user_content = f"""Script: {script}
Timed Captions: {"".join(map(str, captions_timed))}"""
        
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            temperature=1,
            messages=[
                {"role": "system", "content": PROMPTS.get(language, PROMPTS["en"])},
                {"role": "user", "content": user_content}
            ],
            extra_headers={
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost"),
                "X-Title": os.getenv("SITE_NAME", "Video Generator")
            }
        )
        
        # Process the response
        text = response.choices[0].message.content.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Log the response
        log_response(LOG_TYPE_GPT, script, text)
        
        # Parse the JSON response with error handling
        try:
            if isinstance(text, str):
                # Remove JSON code blocks if present
                clean_text = text.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_text)
            return text  # if it's already a dict
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {text}")
            raise ValueError("Invalid JSON response from AI API")
            
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
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