import os 
import requests
from utility.utils import log_response,LOG_TYPE_PEXEL
from utility.retry_utils import retry_api_call, handle_common_errors
import logging
logger = logging.getLogger(__name__)

# PEXELS_API_KEY = os.environ.get('PEXELS_KEY')
PEXELS_API_KEY = "XA4IlmjYKdzM9R7JZX6l4SwVmxTsaJbMvp9l7jf7rE9VVbh5lbxvoKn"

@handle_common_errors
@retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2)
def search_videos(query_string, orientation_landscape=True):
   
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

    response = requests.get(url, headers=headers, params=params)
    json_data = response.json()
    log_response(LOG_TYPE_PEXEL,query_string,response.json())
   
    return json_data


def getBestVideo(query_string, orientation_landscape=True, used_vids=[]):
    try:
        vids = search_videos(query_string, orientation_landscape)
        
        # Check if 'videos' exists in response
        if not vids or 'videos' not in vids:
            logger.error(f"No videos found in response for query: {query_string}")
            return None
            
        videos = vids['videos']

        # Filter videos
        if orientation_landscape:
            filtered_videos = [v for v in videos 
                             if v.get('width', 0) >= 1920 
                             and v.get('height', 0) >= 1080
                             and abs(v.get('width', 0)/v.get('height', 1) - 16/9 < 0.1]
        else:
            filtered_videos = [v for v in videos 
                             if v.get('width', 0) >= 1080 
                             and v.get('height', 0) >= 1920
                             and abs(v.get('height', 0)/v.get('width', 1) - 16/9 < 0.1]

        if not filtered_videos:
            logger.warning(f"No matching videos found for query: {query_string}")
            return None

        # Sort by duration
        sorted_videos = sorted(filtered_videos, key=lambda x: abs(15-int(x.get('duration', 0)))

        # Find best video URL
        for video in sorted_videos:
            for video_file in video.get('video_files', []):
                url = video_file.get('link')
                if url and not (url.split('.hd')[0] in used_vids):
                    return url
                    
        logger.warning(f"No unused videos found for query: {query_string}")
        return None
        
    except Exception as e:
        logger.error(f"Error processing video search for {query_string}: {str(e)}")
        return None

def generate_video_url(timed_video_searches, video_server):
    timed_video_urls = []
    if video_server == "pexel":
        used_links = []
        for (t1, t2), search_terms in timed_video_searches:
            url = None
            # Ensure search_terms is iterable
            if not isinstance(search_terms, (list, tuple)):
                logger.error(f"Expected list of search terms, got: {type(search_terms)}")
                search_terms = []
                
            for query in search_terms:
                try:
                    url = getBestVideo(str(query), orientation_landscape=True, used_vids=used_links)
                    if url:
                        used_links.append(str(url).split('.hd')[0])
                        break
                except Exception as e:
                    logger.error(f"Error processing query {query}: {str(e)}")
                    continue
                    
            timed_video_urls.append([[t1, t2], str(url) if url else None])
    return timed_video_urls