'''
Load pre trained LSTM model and perform inference on MFCC features.
'''
import torch
import logging
import numpy as np
from src.api.utils.utils import log_function, send_update

logging.basicConfig(level=logging.INFO)

@log_function
def load_model(model_path):
    '''
    Load the pre-trained LSTM model from the specified path
    '''
    logging.info(f"Loading model from {model_path}")
    try:
        model = torch.load(model_path)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e
    logging.info(f"Model loaded successfully")
    model.eval() #important for inference
    return model




# main orchestral function
@log_function
async def run_inference(websocket, mfcc_features, model_path):
    '''
    Run model inference on MFCC features and send predictions over websocket
    '''
    logging.info(f"Loading model from {model_path}")
    import os
    if not os.path.exists(model_path):
        # Placeholder: return fake prediction for testing
        logging.warning(f"Model file {model_path} not found. Returning placeholder prediction.")
        num_classes = 4  # e.g., happy, sad, angry, neutral
        fake_probs = [0.1, 0.7, 0.1, 0.1]
        final_class = int(fake_probs.index(max(fake_probs)))
        await send_update(websocket, "processing", {
            "stage": "LSTM_INFERENCE",
            "progress": 100,
            "message": "[Placeholder] Model file not found. Returning fake prediction."
        })
        await send_update(websocket, "completed", {
            "stage": "LSTM_INFERENCE",
            "final_prediction": {
                "class": final_class,
                "confidence": fake_probs
            },
            "message": "[Placeholder] Model inference completed."
        })
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        await send_update(websocket, "error", {"message": f"Error loading model: {e}", "progress": 0})
        return
    
    logging.info(f"Model loaded successfully")

    await send_update(websocket, "processing", {
        "stage": "LSTM_INFERENCE",
        "message": f"Model running..."
    })

    # convert MFCCs to torch tensor with batch dimension
    input_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0) # shape (1, seq_len, num_mfcc)

    hidden = None
    total_frames = input_tensor.shape[1]
    intermediate_predictions = []
    progress = 0
    try:
        for i in range(total_frames):
            frame = input_tensor[:, i:i+1, :] # shape (1, 1, num_mfcc)
            output, hidden = model(frame, hidden) # output shape (1, num_classes)
            probabilities = torch.softmax(output, dim=1).detach().numpy()[0]
            intermediate_predictions.append(probabilities)

            if i % 10 == 0:
                progress = round((i / total_frames) * 100, 2)
                await send_update(websocket, "processing", {
                    "stage": "LSTM_INFERENCE",
                    "progress": progress,
                    "message": f"Processed {i}/{total_frames} frames ({progress}%)",
                    "partial_prediction": probabilities.tolist()
                })
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        await send_update(websocket, "error", {
            "stage": "LSTM_INFERENCE",
            "message": f"Error during inference: {e}",
            "progress": progress
        })
        return

    if not intermediate_predictions:
        logging.error("No predictions were made during inference.")
        await send_update(websocket, "error", {
            "stage": "LSTM_INFERENCE",
            "message": "No predictions were made during inference.",
            "progress": progress
        })
        return

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