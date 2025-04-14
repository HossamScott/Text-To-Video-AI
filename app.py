from openai import OpenAI
import os
import edge_tts
import json
import asyncio
import whisper_timestamped as whisper
from utility.script.script_generator import generate_script
from utility.audio.audio_generator import generate_audio
from utility.captions.timed_captions_generator import generate_timed_captions
from utility.video.background_video_generator import generate_video_url
from utility.render.render_engine import get_output_media
from utility.video.video_search_query_generator import getVideoSearchQueriesTimed, merge_empty_intervals
from flask import Flask, request, jsonify
import uuid
import threading

app = Flask(__name__)

# Global variable to store task status and results
tasks = {}

def generate_video_async(task_id, topic, language, voice, font_settings):
    try:
        SAMPLE_FILE_NAME = f"audio_tts_{task_id}.wav"
        VIDEO_SERVER = "pexel"

        tasks[task_id]['message'] = 'Generating script...'
        response = generate_script(topic, language)
        
        tasks[task_id]['message'] = 'Script generated. Creating audio...'
        asyncio.run(generate_audio(response, SAMPLE_FILE_NAME, voice))
        
        tasks[task_id]['message'] = 'Audio generated. Creating captions...'
        timed_captions = generate_timed_captions(SAMPLE_FILE_NAME)
        
        tasks[task_id]['message'] = 'Captions created. Generating video search terms...'
        search_terms = getVideoSearchQueriesTimed(response, timed_captions)
        
        tasks[task_id]['message'] = 'Searching for background videos...'
        background_video_urls = generate_video_url(search_terms, VIDEO_SERVER) if search_terms else None
        background_video_urls = merge_empty_intervals(background_video_urls)
        
        if background_video_urls:
            tasks[task_id]['message'] = 'Rendering final video...'
            video_path = get_output_media(
                audio_file_path=SAMPLE_FILE_NAME,
                timed_captions=timed_captions,
                background_video_data=background_video_urls,
                video_server=VIDEO_SERVER,
                font_settings=font_settings
            )
            tasks[task_id]['status'] = 'completed'
            tasks[task_id]['result'] = {'video_path': video_path}
        else:
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['error'] = 'No background video available'
            
        if os.path.exists(SAMPLE_FILE_NAME):
            os.remove(SAMPLE_FILE_NAME)
            
    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)


@app.route('/generate', methods=['POST'])
def generate_video():
    data = request.json
    if not data or 'topic' not in data:
        return jsonify({'error': 'Topic is required'}), 400
    
    # Language and voice settings
    language = data.get('language', 'en')
    voice = data.get('voice', 'en-AU-WilliamNeural' if language == 'en' else 'ar-SA-HamedNeural')
    
    # Font settings with defaults
    font_settings = {
        'size': data.get('font_size', 100),
        'color': data.get('font_color', 'white'),
        'stroke_color': data.get('font_stroke_color', 'black'),
        'stroke_width': data.get('font_stroke_width', 3),
        'family': data.get('font_family', 'Arial' if language == 'en' else 'Arial Unicode MS')
    }

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'queued',
        'topic': data['topic'],
        'language': language,
        'settings': {
            'voice': voice,
            'font': font_settings
        },
        'message': 'Waiting to start processing...'
    }
    
    thread = threading.Thread(
        target=generate_video_async, 
        args=(task_id, data['topic'], language, voice, font_settings)
    )
    thread.start()
    
    return jsonify({'task_id': task_id}), 202

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    response = {
        'status': task['status'],
        'message': task.get('message', ''),
        'topic': task['topic'],
        'parameters': {
            'language': task.get('language', 'en'),
            'voice': task['settings'].get('voice', 'en-AU-WilliamNeural'),
            'font_settings': task['settings'].get('font', {
                'size': 100,
                'color': 'white',
                'stroke_color': 'black',
                'stroke_width': 3,
                'family': 'Arial'
            })
        }
    }
    
    if task['status'] == 'completed':
        response['result'] = task['result']
    elif task['status'] == 'failed':
        response['error'] = task['error']
    
    return jsonify(response)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)