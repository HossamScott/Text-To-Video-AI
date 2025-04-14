import os
from openai import OpenAI
import json

# Client initialization outside the function
def get_ai_client():
    """Initialize and return OpenRouter client"""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    ), "google/gemini-2.0-flash-exp:free"  # Default model

# Initialize client and model at module level
client, model = get_ai_client()

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

    extra_params = {}
    if model.startswith("google/"):  # OpenRouter-specific params
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
        json_start_index = content.find('{')
        json_end_index = content.rfind('}')
        print(content)
        content = content[json_start_index:json_end_index+1]
        script = json.loads(content)["script"]
    return script