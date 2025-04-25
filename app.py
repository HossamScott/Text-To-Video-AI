from flask import Flask, request, jsonify
import uuid
import threading
from threading import Lock
import time
import os
import logging
import asyncio
from utility.script.script_generator import generate_script
from utility.audio.audio_generator import generate_audio
from utility.captions.timed_captions_generator import generate_timed_captions
from utility.video.background_video_generator import generate_video_url
from utility.render.render_engine import get_output_media
from utility.video.video_search_query_generator import getVideoSearchQueriesTimed, merge_empty_intervals
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables to store task status and results
tasks = {}
task_lock = Lock()  # Thread-safe lock for tasks dictionary
active_threads = {}  # Dictionary to track running threads

# Add this at the beginning of your existing app.py

@app.route('/tasks', methods=['GET'])
def list_tasks():
    with task_lock:
        simplified_tasks = []
        for task_id, task_data in tasks.items():
            simplified_tasks.append({
                'task_id': task_id,
                'status': task_data['status'],
                'topic': task_data['topic'],
                'progress': task_data.get('progress', 0),
                'created_at': task_data.get('created_at'),
                'updated_at': task_data.get('updated_at'),
                'message': task_data.get('message', '')
            })
        return jsonify({'tasks': simplified_tasks})

@app.route('/tasks/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    with task_lock:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        if tasks[task_id]['status'] not in ['queued', 'processing']:
            return jsonify({'error': 'Task cannot be cancelled in its current state'}), 400
        
        # Mark task for cancellation
        tasks[task_id]['status'] = 'cancelling'
        tasks[task_id]['message'] = 'Cancellation requested'
        tasks[task_id]['updated_at'] = time.time()
        
        # Try to stop the thread if it's running
        if task_id in active_threads:
            thread = active_threads[task_id]
            # This is a gentle way to signal cancellation
            # For more forceful termination, consider using multiprocessing instead
            tasks[task_id]['cancelled'] = True
        
        return jsonify({'status': 'cancellation_requested', 'task_id': task_id})

def update_task_progress(task_id, progress, message=None):
    with task_lock:
        if task_id in tasks:
            tasks[task_id]['progress'] = progress
            if message:
                tasks[task_id]['message'] = message
            tasks[task_id]['updated_at'] = time.time()

def generate_video_async(task_id, topic, language, voice, font_settings):
    try:
        SAMPLE_FILE_NAME = f"audio_tts_{task_id}.wav"
        VIDEO_SERVER = "pexel"
        
        # Check for cancellation before each major step
        def check_cancellation():
            with task_lock:
                if tasks.get(task_id, {}).get('cancelled', False):
                    tasks[task_id]['status'] = 'cancelled'
                    tasks[task_id]['message'] = 'Task was cancelled'
                    tasks[task_id]['updated_at'] = time.time()
                    raise Exception("Task cancelled by user")

        # Initialize task progress
        with task_lock:
            tasks[task_id]['status'] = 'processing'
            tasks[task_id]['progress'] = 0
            tasks[task_id]['created_at'] = time.time()
            tasks[task_id]['updated_at'] = time.time()
            tasks[task_id]['steps'] = [
                'script_generation',
                'audio_generation',
                'caption_generation',
                'video_search',
                'video_rendering'
            ]

        # Step 1: Generate script (10% weight)
        update_task_progress(task_id, 0, 'Generating script...')
        check_cancellation()
        response = generate_script(topic, language)
        update_task_progress(task_id, 10, 'Script generated')
        
        # Step 2: Create audio (20% weight)
        update_task_progress(task_id, 10, 'Script generated. Creating audio...')
        check_cancellation()
        asyncio.run(generate_audio(response, SAMPLE_FILE_NAME, voice))
        update_task_progress(task_id, 30, 'Audio generated')
        
        # Step 3: Create captions (20% weight)
        update_task_progress(task_id, 30, 'Audio generated. Creating captions...')
        check_cancellation()
        timed_captions = generate_timed_captions(SAMPLE_FILE_NAME)
        update_task_progress(task_id, 50, 'Captions created')
        
        # Step 4: Generate video search terms (20% weight)
        update_task_progress(task_id, 50, 'Captions created. Generating video search terms...')
        check_cancellation()
        search_terms = getVideoSearchQueriesTimed(response, timed_captions, language)
        update_task_progress(task_id, 70, 'Search terms generated')
        
        # Step 5: Search for background videos (15% weight)
        update_task_progress(task_id, 70, 'Searching for background videos...')
        check_cancellation()
        background_video_urls = generate_video_url(search_terms, VIDEO_SERVER) if search_terms else None
        background_video_urls = merge_empty_intervals(background_video_urls)
        update_task_progress(task_id, 85, 'Background videos found')
        
        if background_video_urls:
            # Step 6: Render final video (15% weight)
            update_task_progress(task_id, 85, 'Rendering final video...')
            check_cancellation()
            video_path = get_output_media(
                audio_file_path=SAMPLE_FILE_NAME,
                timed_captions=timed_captions,
                background_video_data=background_video_urls,
                video_server=VIDEO_SERVER,
                font_settings=font_settings
            )
            
            with task_lock:
                tasks[task_id]['status'] = 'completed'
                tasks[task_id]['progress'] = 100
                tasks[task_id]['result'] = {'video_path': f'/videos/{video_filename}'}
                tasks[task_id]['message'] = 'Video generation complete'
                tasks[task_id]['updated_at'] = time.time()
        else:
            with task_lock:
                tasks[task_id]['status'] = 'failed'
                tasks[task_id]['error'] = 'No background video available'
                tasks[task_id]['updated_at'] = time.time()
                
        if os.path.exists(SAMPLE_FILE_NAME):
            os.remove(SAMPLE_FILE_NAME)
            
    except Exception as e:
        if str(e) == "Task cancelled by user":
            pass
        else:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"Task {task_id} failed: {error_msg}", exc_info=True)
            with task_lock:
                tasks[task_id]['status'] = 'failed'
                tasks[task_id]['error'] = error_msg
                tasks[task_id]['error_type'] = error_type
                tasks[task_id]['updated_at'] = time.time()
    finally:
        # Clean up the thread reference
        with task_lock:
            if task_id in active_threads:
                del active_threads[task_id]

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
    with task_lock:
        tasks[task_id] = {
            'status': 'queued',
            'topic': data['topic'],
            'language': language,
            'settings': {
                'voice': voice,
                'font': font_settings
            },
            'message': 'Waiting to start processing...',
            'progress': 0,
            'created_at': time.time(),
            'updated_at': time.time(),
            'cancelled': False
        }
    
    thread = threading.Thread(
        target=generate_video_async, 
        args=(task_id, data['topic'], language, voice, font_settings)
    )
    
    with task_lock:
        active_threads[task_id] = thread
    
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'status_url': f'/status/{task_id}',
        'cancel_url': f'/tasks/{task_id}/cancel'
    }), 202

# Update your existing status endpoint
@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    with task_lock:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
        
        task = tasks[task_id]
        response = {
            'task_id': task_id,
            'status': task['status'],
            'progress': task.get('progress', 0),
            'message': task.get('message', ''),
            'topic': task['topic'],
            'created_at': task.get('created_at'),
            'updated_at': task.get('updated_at'),
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
            },
            'links': {
                'cancel': f'/tasks/{task_id}/cancel'
            }
        }
        
        if task['status'] == 'completed':
            response['result'] = task['result']
        elif task['status'] == 'failed':
            response['error'] = {
                'message': task.get('error', 'Unknown error'),
                'type': task.get('error_type', 'unknown'),
                'retryable': task.get('retryable', False),
                'suggestion': task.get('suggestion', 'Please try again later')
            }
        elif task['status'] == 'cancelled':
            response['message'] = 'Task was cancelled by user'
        
        return jsonify(response)

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5050)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        raise