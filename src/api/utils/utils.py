'''
Utility functions for the API module.
'''

import functools
import inspect
import logging

logger = logging.getLogger(__name__)

def log_function(func):
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.info("Entering %s", func.__name__)
            try:
                return await func(*args, **kwargs)
            finally:
                logger.info("Exiting %s", func.__name__)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.info("Entering %s", func.__name__)
            try:
                return func(*args, **kwargs)
            finally:
                logger.info("Exiting %s", func.__name__)
        return sync_wrapper

async def send_update(websocket, status: str, data: dict):
    try:
        await websocket.send_json({
            "status": status,
            "data": data
        })
    except Exception as e:
        logging.error(f"Failed to send update: {e}")
'''
@log_function
def process_order(order_id):
    # process the order
    pass
'''