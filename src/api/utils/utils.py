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

'''
@log_function
def process_order(order_id):
    # process the order
    pass
'''