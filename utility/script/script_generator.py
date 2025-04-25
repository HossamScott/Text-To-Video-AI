import os
import json
import requests
import logging
from utility.retry_utils import retry_api_call, handle_common_errors

logger = logging.getLogger(__name__)


# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")

def get_ai_client():
    """Initialize and return Ollama client configuration"""
    # Test Ollama connection
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        response.raise_for_status()
        return None, OLLAMA_MODEL  # We don't need a client object for Ollama
    except Exception as e:
        logger.error(f"Ollama connection failed: {str(e)}")
        raise ValueError("Failed to connect to Ollama. Is it running?")

# Initialize model
_, model = get_ai_client()

@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2)
def generate_script(topic, language="en"):
    # English prompt
    en_prompt = """
        You're a professional storyteller creating engaging short videos. 
        Create a cohesive narrative with:
        1. Introduction hook (10% of content)
        2. 3-5 connected story points (80%)
        3. Memorable conclusion (10%)
        Maintain:
        - Natural flow between facts
        - Conversational tone
        - Cause/effect relationships
        - Time transitions ("Meanwhile", "Later", etc.)

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

    try:
        # Prepare the Ollama API request
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system", 
                    "content": en_prompt if language == "en" else ar_prompt
                },
                {
                    "role": "user", 
                    "content": topic
                }
            ],
            "stream": False,
            "format": "json"  # Request JSON response
        }

        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        content = response.json()
        message_content = content["message"]["content"]
        
        # Handle the response
        try:
            # Try direct JSON parse first
            if isinstance(message_content, str):
                script_data = json.loads(message_content)
            else:
                script_data = message_content
            
            # Extract the script content
            script = script_data["script"]
            
            # Ensure the script ends with proper punctuation
            if not script[-1] in {'.', '!', '?'}:
                script += '.'
                
            return script
            
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error(f"Failed to parse response. Raw content: {message_content}")
            # Fallback: return the raw content if JSON parsing fails
            return message_content
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {str(e)}")
        raise ValueError("Failed to get response from Ollama")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise