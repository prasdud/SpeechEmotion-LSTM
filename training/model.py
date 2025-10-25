"""
LSTM Model for Emotion Recognition
Compatible with backend frame-by-frame inference
"""
import torch
import torch.nn as nn
from config import *


class EmotionLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, 
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT):
        """
        LSTM model for emotion recognition
        
        Args:
            input_size: Number of MFCC features (13)
            hidden_size: LSTM hidden dimension (256)
            num_layers: Number of LSTM layers (2)
            num_classes: Number of emotion classes (8)
            dropout: Dropout rate (0.3)
        """
        super(EmotionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,  # Input: (batch, seq, features)
            bidirectional=False  # Unidirectional for online inference
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        TRAINING MODE:
            x: (batch, seq_len, input_size) - full sequences
            hidden: None (will be initialized)
        
        INFERENCE MODE (backend):
            x: (batch, 1, input_size) - single timestep
            hidden: (h_n, c_n) from previous timestep or None
        
        Returns:
            output: (batch, num_classes) - logits for current timestep
            hidden: (h_n, c_n) - updated hidden state
        """
        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_size)
        # hidden: tuple of (h_n, c_n), each (num_layers, batch, hidden_size)
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take output from last timestep
        # For training: last timestep of full sequence
        # For inference: the single timestep provided
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        logits = self.fc(last_output)  # (batch, num_classes)
        
        return logits, hidden
    
    def get_num_params(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model():
    """
    Factory function to create model
    """
    model = EmotionLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    )
    
    print(f"Model created:")
    print(f"  Input size: {INPUT_SIZE}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Num classes: {NUM_CLASSES}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Total parameters: {model.get_num_params():,}")
    
    return model


if __name__ == '__main__':
    # Test model creation and forward pass
    model = create_model()
    
    # Test training mode (full sequence)
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, INPUT_SIZE)
    output, hidden = model(x)
    print(f"\nTraining mode test:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Hidden h: {hidden[0].shape}")
    print(f"  Hidden c: {hidden[1].shape}")
    
    # Test inference mode (frame-by-frame)
    print(f"\nInference mode test (frame-by-frame):")
    hidden = None
    for i in range(5):
        frame = torch.randn(1, 1, INPUT_SIZE)  # (batch=1, seq=1, features=13)
        output, hidden = model(frame, hidden)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
        print(f"  Frame {i}: pred={pred.item()}, max_prob={probs.max().item():.3f}")
