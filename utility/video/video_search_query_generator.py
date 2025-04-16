import os
from openai import OpenAI
import json
import re
from datetime import datetime
from utility.utils import log_response, LOG_TYPE_GPT

# Initialize OpenRouter client
def get_ai_client():
    #openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openrouter_key = "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"
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    ), "google/gemini-2.0-flash-exp:free"  # Default model

client, model = get_ai_client()

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
            content = call_OpenAI(script, captions_timed, language).replace("'", '"')
            try:
                out = json.loads(content)
            except Exception as e:
                content = fix_json(content.replace("```json", "").replace("```", ""))
                out = json.loads(content)
        return out
    except Exception as e:
        print("Error processing response:", e)
    return None

def call_OpenAI(script, captions_timed, language="en"):
    user_content = f"""Script: {script}
Timed Captions: {"".join(map(str, captions_timed))}"""
    
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
    
    text = response.choices[0].message.content.strip()
    text = re.sub(r'\s+', ' ', text)
    log_response(LOG_TYPE_GPT, script, text)
    return text

def merge_empty_intervals(segments):
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