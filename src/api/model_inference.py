'''
Load pre trained LSTM model and perform inference on MFCC features.
'''
import torch
import logging
import numpy as np
from src.api.utils.utils import log_function, send_update
from src.api.model import EmotionLSTM

logging.basicConfig(level=logging.INFO)

@log_function
def load_model(model_path):
    '''
    Load the pre-trained LSTM model from the specified path
    '''
    logging.info(f"Loading model from {model_path}")
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Extract config
        config = checkpoint['model_config']
        
        # Reconstruct model architecture WITH attention and batch norm
        model = EmotionLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            use_attention=config.get('use_attention', False),  # CRITICAL: Enable attention!
            use_batch_norm=config.get('use_batch_norm', False)  # CRITICAL: Enable batch norm!
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Important for inference
        
        logging.info(f"Model loaded successfully (epoch {checkpoint['epoch']}, val_acc {checkpoint['val_acc']:.2f}%)")
        logging.info(f"Model config: input_size={config['input_size']}, attention={config.get('use_attention')}, batch_norm={config.get('use_batch_norm')}")
        
        # Return model and emotion labels
        return model, checkpoint.get('emotion_labels', {
            0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
            4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprised"
        })
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e




# main orchestral function
@log_function
async def run_inference(websocket, mfcc_features, model_path):
    '''
    Run model inference on MFCC features and send predictions over websocket
    '''
    logging.info(f"Loading model from {model_path}")
    try:
        model, emotion_labels = load_model(model_path)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        await send_update(websocket, "error", {"message": f"Error loading model: {e}"})
        return
    
    logging.info(f"Model loaded successfully")

    await send_update(websocket, "processing", {
        "stage": "LSTM_INFERENCE",
        "message": f"Model running..."
    })

    # convert MFCCs to torch tensor with batch dimension
    input_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0) # shape (1, seq_len, num_mfcc)
    
    total_frames = input_tensor.shape[1]
    seq_length = torch.tensor([total_frames])  # Actual sequence length for attention
    
    try:
        with torch.no_grad():  # No gradients needed for inference
            # BATCH MODE INFERENCE (like training) - critical for attention mechanism!
            # Process entire sequence at once instead of frame-by-frame
            output, _ = model(input_tensor, hidden=None, lengths=seq_length)
            
            # Get probabilities
            final_probabilities = torch.softmax(output, dim=1).detach().numpy()[0]
            final_class = int(np.argmax(final_probabilities))
            final_emotion = emotion_labels[final_class]
            
            # Send progress updates
            await send_update(websocket, "processing", {
                "stage": "LSTM_INFERENCE",
                "progress": 50,
                "message": f"Processing full sequence of {total_frames} frames...",
                "partial_prediction": {
                    "probabilities": final_probabilities.tolist(),
                    "emotion": final_emotion,
                    "class": final_class
                }
            })
            
            await send_update(websocket, "processing", {
                "stage": "LSTM_INFERENCE",
                "progress": 100,
                "message": f"Inference complete - Predicted: {final_emotion}",
                "partial_prediction": {
                    "probabilities": final_probabilities.tolist(),
                    "emotion": final_emotion,
                    "class": final_class
                }
            })
            
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        await send_update(websocket, "error", {
            "stage": "LSTM_INFERENCE",
            "message": f"Error during inference: {e}"
        })
        return
    
    logging.info(f"Final prediction: {final_emotion} (class {final_class}) with confidence {final_probabilities[final_class]:.2%}")
    
    await send_update(websocket, "completed", {
        "stage": "LSTM_INFERENCE",
        "progress": 100,
        "final_prediction": {
            "class": final_class,
            "emotion": final_emotion,
            "confidence": final_probabilities.tolist(),  # Send full array for confidence distribution
            "probabilities": final_probabilities.tolist(),
            "all_emotions": {emotion_labels[i]: float(final_probabilities[i]) for i in range(len(final_probabilities))}
        },
        "message": f"Model inference completed. Predicted emotion: {final_emotion}"
    })
    
    # Send final "completed" stage to mark pipeline as fully done
    await send_update(websocket, "completed", {
        "stage": "completed",
        "progress": 100,
        "message": "Pipeline completed successfully!"
    })