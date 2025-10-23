'''
Initializes the FastAPI application, set up the websocket endpoint, handles websocket connections.
'''
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # Echo the received message back to the client
        await websocket.send_text(f"Message received: {data}")

