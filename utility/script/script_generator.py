import os
from openai import OpenAI
import json
from utility.retry_utils import retry_api_call, handle_common_errors

def get_ai_client():    
    openrouter_key = "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"
    if len(openrouter_key) > 30:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        ), "google/gemini-2.0-flash-exp:free"

    openai_key = "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"
    if len(openai_key) > 30:
        return OpenAI(api_key=openai_key), "gpt-4o"
    
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if len(groq_api_key) > 30:
        return Groq(api_key=groq_api_key), "mixtral-8x7b-32768"
    
    raise ValueError("No valid API key found in environment variables")

# Initialize client and model at module level
client, model = get_ai_client()

@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2)
def generate_script(topic, language="en"):
    # English prompt
    en_prompt = """
        You are a seasoned content writer for a YouTube Shorts channel, specializing in facts videos. 
        Your facts shorts are concise, each lasting less than 50 seconds (approximately 140 words). 
        They are incredibly engaging and original. When a user requests a specific type of facts short, you will create it.

        For instance, if the user asks for:
        Weird facts
        You would produce content like this:

        Weird facts you don't know:
        - Bananas are berries, but strawberries aren't.
        - A single cloud can weigh over a million pounds.
        - There's a species of jellyfish that is biologically immortal.
        - Honey never spoils; archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible.
        - The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.
        - Octopuses have three hearts and blue blood.

        You are now tasked with creating the best short script based on the user's requested type of 'facts'.

        Keep it brief, highly interesting, and unique.

        Strictly output the script in a JSON format like below, and only provide a parsable JSON object with the key 'script'.

        {"script": "Here is the script ..."}
        """

    # Arabic prompt
    ar_prompt = """
        أنت كاتب محتوى محترف لقناة يوتيوب شورتس، متخصص في مقاطع الفيديو الحقائقية.
        مقاطعك القصيرة تكون مختصرة، كل منها أقل من 50 ثانية (حوالي 140 كلمة).
        يجب أن تكون جذابة وأصلية. عندما يطلب المستخدم نوعًا معينًا من الحقائق، ستقوم بإنشائه.

        على سبيل المثال، إذا طلب المستخدم:
        حقائق غريبة
        ستنشئ محتوى مثل:

        حقائق غريبة لا تعرفها:
        - الموز من التوت لكن الفراولة ليست كذلك.
        - يمكن أن يزن السحاب الواحد أكثر من مليون رطل.
        - هناك نوع من قناديل البحر خالد بيولوجيًا.
        - العسل لا يفسد؛ فقد عثر علماء الآثار على أواني عسل في مقابر مصرية عمرها أكثر من 3000 سنة ولا تزال صالحة للأكل.
        - أقصر حرب في التاريخ كانت بين بريطانيا وزنجبار في 27 أغسطس 1896. استسلمت زنجبار بعد 38 دقيقة.
        - للأخطبوط ثلاثة قلوب ودم أزرق.

        مهمتك الآن هي إنشاء أفضل نص قصير بناءً على نوع "الحقائق" المطلوب من المستخدم.

        اجعله مختصرًا، مثيرًا للاهتمام وفريدًا من نوعه.

        أخرج النص بتنسيق JSON كما هو موضح أدناه، وقدم فقط كائن JSON قابل للتحليل بمفتاح 'script'.

        {"script": "هنا النص ..."}
        """

    # Select prompt based on language
    prompt = en_prompt if language == "en" else ar_prompt

    # Prepare extra parameters if using OpenRouter
    extra_params = {}
    if "openrouter.ai" in getattr(client, "base_url", ""):
        extra_params = {
            "extra_headers": {
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost"),
                "X-Title": os.getenv("SITE_NAME", "Video Generator")
            }
        }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": topic}
        ],
        **extra_params
    )
    
    content = response.choices[0].message.content
    try:
        script = json.loads(content)["script"]
    except Exception as e:
        # Fallback JSON extraction if direct parsing fails
        json_start_index = content.find('{')
        json_end_index = content.rfind('}')
        print(f"JSON parsing error, attempting fallback extraction. Original content: {content}")
        content = content[json_start_index:json_end_index+1]
        script = json.loads(content)["script"]
    return script