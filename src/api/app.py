'''
Initializes the FastAPI application, set up the websocket endpoint, handles websocket connections.
'''
from fastapi import FastAPI, WebSocket
import logging
from src.api.websocket_handler import handle_websocket
from src.api.utils.utils import log_function

app = FastAPI()
logging.basicConfig(level=logging.INFO)

@log_function
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logging.info("WebSocket connection request received.")
    try:
        await websocket.accept()
        logging.info("WebSocket connection accepted.")
        logging.info("Starting WebSocket handler.")
        await handle_websocket(websocket)
    except Exception as e:
        logging.error(f"WebSocket handler error: {e}", exc_info=True)
        await websocket.close()

    finally:
        logging.info("WebSocket handler finished.")