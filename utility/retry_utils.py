import time
import logging
from functools import wraps
from requests.exceptions import RequestException
from openai import APIError, RateLimitError, APIConnectionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2):
    """
    Decorator for retrying API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Factor by which delay increases each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                
                except RateLimitError as e:
                    logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds... (Attempt {retries + 1}/{max_retries})")
                    time.sleep(delay)
                    retries += 1
                    delay *= backoff_factor
                    
                except APIConnectionError as e:
                    logger.warning(f"API connection error. Retrying in {delay} seconds... (Attempt {retries + 1}/{max_retries})")
                    time.sleep(delay)
                    retries += 1
                    delay *= backoff_factor
                    
                except APIError as e:
                    logger.error(f"API error: {str(e)}")
                    raise
                    
                except RequestException as e:
                    logger.error(f"Request failed: {str(e)}")
                    raise
                    
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    raise
                    
            # If we've exhausted retries
            raise Exception(f"Max retries ({max_retries}) exceeded for API call")
        return wrapper
    return decorator

def handle_common_errors(func):
    """
    Decorator for handling common API errors with user-friendly messages
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
            
        except RateLimitError:
            raise Exception("Our service is currently experiencing high demand. Please try again shortly.")
            
        except APIConnectionError:
            raise Exception("Unable to connect to the service. Please check your internet connection.")
            
        except APIError as e:
            if e.status_code == 401:
                raise Exception("Authentication failed. Please check your API keys.")
            elif e.status_code == 403:
                raise Exception("Permission denied. Please check your account permissions.")
            elif e.status_code == 404:
                raise Exception("Requested resource not found.")
            elif e.status_code == 429:
                raise Exception("Too many requests. Please wait before trying again.")
            else:
                raise Exception(f"API error occurred: {str(e)}")
                
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")
            
    return wrapper