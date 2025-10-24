'''
Utility functions for the API module.
'''

import logging
logger = logging.getLogger(__name__)

def log_function(func):
    def wrapper(*args, **kwargs):
        logger.info("Entering %s", func.__name__)
        result = func(*args, **kwargs)
        logger.info("Exiting %s", func.__name__)
        return result
    return wrapper

async def send_update(websocket, status: str, data: dict):
    '''
        format JSON message and send over websocket
    '''
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