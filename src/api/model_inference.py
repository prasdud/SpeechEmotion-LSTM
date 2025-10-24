'''
Load pre trained LSTM model and perform inference on MFCC features.
'''
import torch
from src.api.websocket_handler import send_update
import logging
import numpy as np


def load_model(model_path):
    '''
    Load the pre-trained LSTM model from the specified path
    '''
    model = torch.load(model_path)
    model.eval() #important for inference
    return model




# main orchestral function
async def run_inference(websocket, mfcc_features, model_path):
    '''
    Run model inference on MFCC features and send predictions over websocket
    '''
    logging.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    logging.info(f"Model loaded successfully")

    await send_update(websocket, "processing", {
        "stage": "LSTM_INFERENCE",
        "message": f"Model running..."
    })

    # convert MFCCs to toruch tensor with batch dimension
    input_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0) # shape (1, seq_len, num_mfcc)

    hidden = None
    total_frames = input_tensor.shape[1]
    intermediate_predictions = []
    for i in range(total_frames):
        frame = input_tensor[:, i:i+1, :] # shape (1, 1, num_mfcc)

        output, hidden = model(frame, hidden) # output shape (1, num_classes)
        probabilities = torch.softmax(output, dim=1).detach().numpy()[0]

        intermediate_predictions.append(probabilities)

        if i % 10 == 0: # send update every 10 frames
            progress = round((i / total_frames) * 100, 2)
            await send_update(websocket, "processing", {
                "stage": "LSTM_INFERENCE",
                "progress": progress,
                "message": f"Processed {i}/{total_frames} frames ({progress}%)",
                "partial_prediction": probabalities.tolist()
            })

    # final prediction from last frame
    final_probabilities = intermediate_predictions[-1]
    final_class = int(np.argmax(final_probabilities))
    
    '''
    test out instead of last frame for final pred, averale all intermediate preds
    final_probabilities = np.mean(intermediate_predictions, axis=0)
    final_class = int(np.argmax(final_probabilities))
    '''
    
    logging.info(f"Final prediction: class {final_class} with probabilities {final_probabilities}")
    
    await send_update(websocket, "completed", {
        "stage": "LSTM_INFERENCE",
        "final_prediction": {
            "class": final_class,
            "confidence": final_probabilities.tolist()
        },
        "message": "Model inference completed."
    })