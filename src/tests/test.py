import asyncio
import websockets
import json
import base64

ACTIONS = ["upload_audio", "mfcc_extraction", "model_inference"]

async def test_interactive_pipeline():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Step 1: Upload audio
        with open("sample.wav", "rb") as f:
            file_bytes = f.read()
        base64_str = base64.b64encode(file_bytes).decode("utf-8")
        
        upload_msg = {
            "action": "upload_audio",
            "data": {
                "filename": "sample.wav",
                "content": base64_str
            }
        }
        await websocket.send(json.dumps(upload_msg))
        print("Sent upload_audio")

        # Listen and print updates while processing
        while True:
            response = await websocket.recv()
            response_data = json.loads(response)
            print("Received:", json.dumps(response_data, indent=2))

            # Extract stage and message from 'data' if present
            data = response_data.get("data", {})
            stage = data.get("stage")
            message = data.get("message", "")
            final_prediction = data.get("final_prediction") or response_data.get("final_prediction")

            # After upload_audio stage is complete, trigger MFCC extraction
            if (
                (stage == "audio_processing" and message == "Audio framed") or
                (stage == "AUDIO_UPLOAD")
            ):
                mfcc_msg = {
                    "action": "mfcc_extraction",
                    "data": {}  # the backend already has frames
                }
                await websocket.send(json.dumps(mfcc_msg))
                print("Sent mfcc_extraction")

            # After MFCC extraction stage is complete, trigger model inference
            if (
                (stage == "MFCC_EXTRACTION" and message == "MFCC extraction completed") or
                (stage == "MFCC_EXTRACTION" and "done" in message.lower())
            ):
                inference_msg = {
                    "action": "model_inference",
                    "data": {}  # the backend already has MFCCs
                }
                await websocket.send(json.dumps(inference_msg))
                print("Sent model_inference")

            # Stop listening after final prediction
            if (stage == "LSTM_INFERENCE" and final_prediction):
                print("Pipeline completed.")
                break

asyncio.run(test_interactive_pipeline())
