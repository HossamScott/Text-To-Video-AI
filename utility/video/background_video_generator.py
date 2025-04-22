import os 
import requests
from utility.utils import log_response, LOG_TYPE_PEXEL
from utility.retry_utils import retry_api_call, handle_common_errors
import logging

logger = logging.getLogger(__name__)

# Use environment variable for API key
PEXELS_API_KEY = os.environ.get('PEXELS_KEY', "XA4IlmjYKdzM9R7JZX6l4SwVmxTsaJbMvp9l7jf7rE9VVbh5lbxvoKn")

@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2)
def search_videos(query_string, orientation_landscape=True):
    """Search for videos on Pexels with proper error handling"""
    url = "https://api.pexels.com/videos/search"
    headers = {
        "Authorization": PEXELS_API_KEY,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {
        "query": query_string,
        "orientation": "landscape" if orientation_landscape else "portrait",
        "per_page": 15
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    json_data = response.json()
    log_response(LOG_TYPE_PEXEL, query_string, json_data)
    return json_data

def getBestVideo(query_string, orientation_landscape=True, used_vids=[]):
    """Get the best matching video with enhanced error handling"""
    try:
        vids = search_videos(query_string, orientation_landscape)
        
        # Check if response contains videos
        if not vids or 'videos' not in vids or not vids['videos']:
            logger.warning(f"No videos found for query: {query_string}")
            return None
            
        videos = vids['videos']

        # Filter videos by resolution and aspect ratio
        if orientation_landscape:
            filtered_videos = [
                video for video in videos 
                if video.get('width', 0) >= 1920 
                and video.get('height', 0) >= 1080
                and abs(video.get('width', 0)/video.get('height', 1) - 16/9) < 0.1  # Allow slight aspect ratio variation
            ]
        else:
            filtered_videos = [
                video for video in videos 
                if video.get('width', 0) >= 1080 
                and video.get('height', 0) >= 1920
                and abs(video.get('height', 0)/video.get('width', 1) - 16/9) < 0.1
            ]

        if not filtered_videos:
            logger.warning(f"No matching videos after filtering for: {query_string}")
            return None

        # Sort by duration closest to 15 seconds
        sorted_videos = sorted(filtered_videos, key=lambda x: abs(15 - x.get('duration', 0)))

        # Find the first usable video URL
        for video in sorted_videos:
            for video_file in video.get('video_files', []):
                try:
                    if orientation_landscape:
                        if video_file.get('width') == 1920 and video_file.get('height') == 1080:
                            url = video_file.get('link')
                            if url and not any(url.startswith(used) for used in used_vids):
                                return url
                    else:
                        if video_file.get('width') == 1080 and video_file.get('height') == 1920:
                            url = video_file.get('link')
                            if url and not any(url.startswith(used) for used in used_vids):
                                return url
                except Exception as e:
                    logger.warning(f"Error processing video file: {str(e)}")
                    continue

        logger.warning(f"No unused videos found for: {query_string}")
        return None

    except Exception as e:
        logger.error(f"Video search failed for {query_string}: {str(e)}")
        return None

def generate_video_url(timed_video_searches, video_server):
    """Generate video URLs with improved error handling"""
    timed_video_urls = []
    if video_server == "pexel":
        used_links = []
        for (t1, t2), search_terms in timed_video_searches:
            url = None
            if not isinstance(search_terms, (list, tuple)):
                search_terms = [search_terms] if search_terms else []
                
            for query in search_terms:
                try:
                    url = getBestVideo(str(query), True, used_links)
                    if url:
                        used_links.append(url.split('.hd')[0])
                        break
                except Exception as e:
                    logger.error(f"Error processing query {query}: {str(e)}")
                    continue
                    
            timed_video_urls.append([[t1, t2], url if url else None])
    
    return timed_video_urls