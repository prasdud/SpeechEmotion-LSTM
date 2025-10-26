"""
Enhanced LSTM model with attention mechanism and batch normalization
"""
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    """
    Attention mechanism to focus on important timesteps
    """
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output, lengths=None):
        """
        lstm_output: (batch, seq_len, hidden_size)
        lengths: (batch,) - actual sequence lengths
        
        Returns:
            context: (batch, hidden_size) - weighted sum of lstm_output
            attention_weights: (batch, seq_len) - attention scores
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, seq_len)
        
        # Mask padding if lengths provided
        if lengths is not None:
            # Ensure lengths is on the same device as lstm_output
            lengths = lengths.to(lstm_output.device)
            mask = torch.arange(lstm_output.size(1), device=lstm_output.device)[None, :] < lengths[:, None]
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)  # (batch, 1, hidden)
        context = context.squeeze(1)  # (batch, hidden)
        
        return context, attention_weights


class EmotionLSTM(nn.Module):
    """
    Enhanced LSTM model for emotion recognition from MFCC features
    
    Improvements over baseline:
    - Attention mechanism to focus on emotional cues
    - Batch normalization for training stability
    - Higher dropout (0.5) to reduce overfitting
    - Support for 39 features (13 MFCCs + deltas + delta-deltas)
    
    Supports two modes:
    1. Training/Batch mode: Process full sequences with PackedSequence
    2. Inference mode: Frame-by-frame streaming inference
    """
    def __init__(self, input_size=39, hidden_size=256, num_layers=2, num_classes=8, dropout=0.5,
                 use_attention=True, use_batch_norm=True):
        super(EmotionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Keep unidirectional for real-time compatibility
        )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = AttentionLayer(hidden_size)
        
        # Batch normalization (optional)
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, hidden=None, lengths=None):
        """
        Forward pass with support for variable-length sequences
        
        TRAINING MODE:
            x: (batch, seq_len, input_size) - padded sequences
            hidden: None (will be initialized)
            lengths: (batch,) - actual sequence lengths (for packing)
        
        INFERENCE MODE (backend):
            x: (batch, 1, input_size) - single timestep
            hidden: (h_n, c_n) from previous timestep or None
            lengths: None (not needed for single timestep)
        
        Returns:
            output: (batch, num_classes) - logits for current timestep
            hidden: (h_n, c_n) - updated hidden state
        """
        # Pack sequences if lengths provided (training mode)
        if lengths is not None:
            # Pack padded sequences to skip padding during LSTM processing
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, hidden = self.lstm(packed_input, hidden)
            
            # Unpack sequences
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            # Inference mode - no packing needed
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Get the last valid output for each sequence
        if lengths is not None and self.use_attention:
            # Use attention mechanism to focus on important timesteps
            last_output, attention_weights = self.attention(lstm_out, lengths)
        elif lengths is not None:
            # Use actual sequence lengths to get last valid output
            batch_size = lstm_out.size(0)
            idx = (lengths - 1).long().to(lstm_out.device)
            last_output = lstm_out[range(batch_size), idx, :]  # (batch, hidden_size)
        else:
            # Inference mode - just take last timestep
            last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm and last_output.size(0) > 1:  # Batch norm needs batch_size > 1
            last_output = self.batch_norm(last_output)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Fully connected layer
        logits = self.fc(last_output)  # (batch, num_classes)
        
        return logits, hidden
