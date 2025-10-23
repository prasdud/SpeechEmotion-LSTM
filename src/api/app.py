'''
Initializes the FastAPI application, set up the websocket endpoint, handles websocket connections.
'''
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import logging
from src.api.websocket_handler import handle_websocket

app = FastAPI()
logging.basicConfig(level=logging.INFO)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await handle_websocket(websocket)
    # while True:
    #     data = await websocket.receive_text()
    #     # Echo the received message back to the client
    #     logging.info(f"Received message: {data}")
    #     await websocket.send_text(f"Message received: {data}")

