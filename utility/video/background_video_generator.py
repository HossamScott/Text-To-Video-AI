import os 
import requests
from utility.utils import log_response, LOG_TYPE_PEXEL
from utility.retry_utils import retry_api_call, handle_common_errors
import logging

logger = logging.getLogger(__name__)

# Use environment variable for API key
PEXELS_API_KEY = os.environ.get('PEXELS_KEY', "aXA4IlmjYKdzM9R7JZX6l4SwVmxTsaJbMvp9l7jf7rE9VVbh5lbxvoKn")

@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2)
def search_videos(query_string, orientation_landscape=True):
    """Search for videos on Pexels with enhanced error handling"""
    if not query_string or len(query_string.strip()) < 2:
        logger.warning(f"Invalid query string: {query_string}")
        return None

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

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        json_data = response.json()
        log_response(LOG_TYPE_PEXEL, query_string, json_data)
        return json_data
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            logger.error("Invalid Pexels API key - update PEXELS_KEY environment variable")
        return None

def getBestVideo(query_string, orientation_landscape=True, used_vids=None):
    """Get the best matching video with improved query handling"""
    used_vids = used_vids or []
    
    try:
        vids = search_videos(query_string, orientation_landscape)
        if not vids or 'videos' not in vids or not vids['videos']:
            return None
            
        videos = vids['videos']
        min_width = 1920 if orientation_landscape else 1080
        min_height = 1080 if orientation_landscape else 1920

        filtered_videos = [
            video for video in videos 
            if video.get('width', 0) >= min_width 
            and video.get('height', 0) >= min_height
            and video.get('id') not in used_vids
        ]

        if not filtered_videos:
            return None

        # Prioritize videos closest to 15 seconds
        sorted_videos = sorted(filtered_videos, 
                             key=lambda x: abs(15 - x.get('duration', 0)))

        # Find first usable video URL
        for video in sorted_videos:
            video_files = sorted(video.get('video_files', []),
                             key=lambda x: x.get('width', 0), reverse=True)
            
            for video_file in video_files:
                if (video_file.get('width') == min_width and 
                    video_file.get('height') == min_height):
                    return video_file.get('link'), video.get('id')

        return None
    except Exception as e:
        logger.error(f"Video search failed for {query_string}: {str(e)}")
        return None

def generate_video_url(timed_video_searches, video_server):
    """Generate video URLs with proper query handling"""
    if video_server != "pexel":
        return []

    timed_video_urls = []
    used_video_ids = []
    
    for (t1, t2), search_terms in timed_video_searches:
        url, vid_id = None, None
        
        # Create proper search queries
        queries = []
        if isinstance(search_terms, (list, tuple)) and len(search_terms) >= 3:
            # Try combination of first 3 keywords
            combined_query = " ".join(str(kw) for kw in search_terms[:3])
            queries.append(combined_query)
            # Add individual keywords as fallback
            queries.extend(search_terms[:3])
        else:
            queries.append(str(search_terms))

        for query in queries:
            try:
                result = getBestVideo(query, True, used_video_ids)
                if result:
                    url, vid_id = result
                    used_video_ids.append(vid_id)
                    break
            except Exception as e:
                logger.error(f"Query {query} failed: {str(e)}")
                continue
                
        timed_video_urls.append([[t1, t2], url if url else None])
    
    return timed_video_urls