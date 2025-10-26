"""
Central hub for all messages from backend to frontend and vice versa.
Handles per-connection state for audio frames, MFCCs, and inference results.
"""
import json
import logging
import base64
import os

from src.api.audio_processing import process_audio
from src.api.mfcc_extraction import compute_mfcc
from src.api.model_inference import run_inference
from src.api.utils.utils import send_update, log_function

logging.basicConfig(level=logging.INFO)

@log_function
async def handle_websocket(websocket):
    """
    Main websocket handler
    """
    state = {}  # store frames, sample_rate, MFCCs, etc. for this connection

    try:
        while True:
            raw_data = await websocket.receive_text()
            logging.info(f"Received data: {raw_data}")

            try:
                message = json.loads(raw_data)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON: {e}")
                await send_update(websocket, "error", {"message": f"Error decoding JSON: {e}"})
                continue

            logging.info(f"Parsed message: {message}")
            await handle_message(websocket, message, state)

    except Exception as e:
        logging.error(f"Websocket error: {e}")
        await websocket.close()


@log_function
async def handle_message(websocket, message, state):
    """
    Handle messages from frontend and call corresponding backend functions
    Uses 'state' to store intermediate results per connection
    """
    action = message.get("action")
    data = message.get("data", {})

    logging.info(f"Handling message with action: {action}")

    if action == "upload_audio":
        logging.info("Processing audio upload.")
        try:
            file_bytes = base64.b64decode(data["content"])
        except Exception as e:
            await send_update(websocket, "error", {"message": f"Failed to decode audio: {e}"})
            return

        frames, sr = await process_audio(file_bytes, websocket)
        state["frames"] = frames
        state["sr"] = sr
        await send_update(websocket, "completed", {
            "stage": "AUDIO_UPLOAD",
            "progress": 25,
            "message": f"Audio uploaded and framed into {frames.shape[1]} frames"
        })

    elif action == "mfcc_extraction":
        frames = state.get("frames")
        sr = state.get("sr")
        if frames is None or sr is None:
            await send_update(websocket, "error", {"message": "No audio frames found. Upload audio first."})
            return

        logging.info("Processing MFCC extraction.")
        mfccs = await compute_mfcc(websocket, frames, sample_rate=sr, num_mfcc=13)
        state["mfccs"] = mfccs
        await send_update(websocket, "completed", {
            "stage": "MFCC_EXTRACTION",
            "progress": 50,
            "message": f"MFCC extraction done. Shape: {mfccs.shape}"
        })

    elif action == "model_inference":
        mfccs = state.get("mfccs")
        if mfccs is None:
            await send_update(websocket, "error", {"message": "No MFCCs found. Extract MFCC first."})
            return

        logging.info("Processing model inference.")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = data.get("model_path", os.path.join(current_dir, "model.pth"))
        await run_inference(websocket, mfccs, model_path)

    else:
        logging.warning(f"Unknown action: {action}")
        await send_update(websocket, "error", {"message": f"Unknown action: {action}"})
