import asyncio
import websockets
import json
import base64

async def test_audio_upload():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Simulate uploading audio
        with open("sample.wav", "rb") as f:
            file_bytes = f.read()
        
        # Encode to base64 and decode to string
        base64_str = base64.b64encode(file_bytes).decode("utf-8") # json dumps requires str, not bytes
        
        await websocket.send(json.dumps({
            "action": "upload_audio",
            "data": {
                "filename": "sample.wav",
                "content": base64_str
            }
        }))

        # Listen for messages from backend
        while True:
            response = await websocket.recv()
            print("Received:", response)

asyncio.run(test_audio_upload())
