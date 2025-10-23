'''
Central hub for all messages from backend to frontend and vice versa.
'''
import json
import logging
from audio_processing import process_audio
from mfcc_extraction import compute_mfcc
from model_inference import run_inference

logging.basicConfig(level=logging.INFO)

async def handle_websocket(websocket):
    '''
        main websocket handler
        calls handle_message to recieve messages
        calls send_update to send messages
    '''
    try:
        while True:
            raw_data = await websocket.receive_text()
            logging.info(f"Received data: {raw_data}")
            message = json.loads(raw_data)
            logging.info(f"Parsed message: {message}")
            await handle_message(websocket, message)
    except Exception as e:
        logging.error(f"Websocket error: {e}")
        await websocket.close()
        
async def handle_message(websocket, message):
    ''''
        recieve messages from frontend
        parse JSON 'action' field
        call corresponding function in backend
        send status updates / final prediction
    '''
    action = message.get("action")
    data = message.get("data", {})

    if action == "upload_audio":
        await process_audio(websocket, data)

    elif action == "mfcc_extraction":
        await compute_mfcc(websocket, data)

    elif action == "model_inference":
        await run_inference(websocket, data)
    else:
        logging.warning(f"Unknown action: {action}")
        await send_update(websocket, "error", {"message": f"Unknown action: {action}"})


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