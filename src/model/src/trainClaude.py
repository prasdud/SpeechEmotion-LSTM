"""
Training script for speech emotion recognition models
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from features import AudioFeatureExtractor
from models import SERModelBuilder, get_callbacks


class SERTrainer:
    """Train speech emotion recognition models"""
    
    def __init__(
        self,
        feature_type: str = 'mfcc',
        n_mfcc: int = 40,
        max_time_steps: int = 174,
        model_type: str = 'bilstm'
    ):
        """
        Args:
            feature_type: Type of features to extract
            n_mfcc: Number of MFCC coefficients
            max_time_steps: Maximum time steps for padding
            model_type: Type of model to train
        """
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc
        self.max_time_steps = max_time_steps
        self.model_type = model_type
        
        self.extractor = AudioFeatureExtractor(n_mfcc=n_mfcc)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def extract_features_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from all files in dataframe
        
        Args:
            df: DataFrame with file paths
            
        Returns:
            Feature array (n_samples, time_steps, n_features)
        """
        features_list = []
        valid_indices = []
        
        print(f"Extracting {self.feature_type} features...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            features = self.extractor.extract_all_features(
                row['file_path'],
                feature_type=self.feature_type
            )
            
            if features is not None:
                # Pad or truncate to fixed length
                features = self.extractor.pad_or_truncate(features, self.max_time_steps)
                features_list.append(features)
                valid_indices.append(idx)
        
        # Stack features
        features_array = np.stack(features_list, axis=0)
        
        # Transpose to (n_samples, time_steps, n_features)
        features_array = np.transpose(features_array, (0, 2, 1))
        
        print(f"Features shape: {features_array.shape}")
        
        return features_array, valid_indices
    
    def prepare_data(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str
    ):
        """
        Prepare training data
        
        Args:
            train_csv: Path to train metadata CSV
            val_csv: Path to validation metadata CSV
            test_csv: Path to test metadata CSV
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Load metadata
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Extract features
        X_train, train_valid_idx = self.extract_features_from_dataframe(train_df)
        X_val, val_valid_idx = self.extract_features_from_dataframe(val_df)
        X_test, test_valid_idx = self.extract_features_from_dataframe(test_df)
        
        # Get corresponding labels
        y_train = train_df.iloc[train_valid_idx]['emotion'].values
        y_val = val_df.iloc[val_valid_idx]['emotion'].values
        y_test = test_df.iloc[test_valid_idx]['emotion'].values
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # One-hot encode
        y_train_cat = to_categorical(y_train_encoded)
        y_val_cat = to_categorical(y_val_encoded)
        y_test_cat = to_categorical(y_test_encoded)
        
        # Normalize features
        print("\nNormalizing features...")
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_val_scaled = self.scaler.transform(X_val_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        print(f"\nFinal shapes:")
        print(f"X_train: {X_train_scaled.shape}, y_train: {y_train_cat.shape}")
        print(f"X_val: {X_val_scaled.shape}, y_val: {y_val_cat.shape}")
        print(f"X_test: {X_test_scaled.shape}, y_test: {y_test_cat.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_cat, y_val_cat, y_test_cat
    
    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int = 100,
        batch_size: int = 32,
        model_save_path: str = 'experiments/best_model.h5'
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            model_save_path: Path to save best model
            
        Returns:
            model, history
        """
        # Create output directory
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = y_train.shape[1]
        
        print(f"\nBuilding {self.model_type} model...")
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")
        
        if self.model_type == 'rnn':
            model = SERModelBuilder.build_simple_rnn(input_shape, num_classes)
        elif self.model_type == 'lstm':
            model = SERModelBuilder.build_lstm(input_shape, num_classes)
        elif self.model_type == 'bilstm':
            model = SERModelBuilder.build_bidirectional_lstm(input_shape, num_classes)
        elif self.model_type == 'gru':
            model = SERModelBuilder.build_gru(input_shape, num_classes)
        elif self.model_type == 'cnn_lstm':
            model = SERModelBuilder.build_cnn_lstm(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model.summary()
        
        # Get callbacks
        callback_list = get_callbacks(model_save_path)
        
        # Train model
        print(f"\nTraining {self.model_type} model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        return model, history
    
    def save_artifacts(self, save_dir: str = 'experiments'):
        """
        Save preprocessing artifacts
        
        Args:
            save_dir: Directory to save artifacts
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save label encoder
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
        print(f"Label encoder saved to {save_dir}/label_encoder.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        print(f"Scaler saved to {save_dir}/scaler.pkl")
        
        # Save config
        config = {
            'feature_type': self.feature_type,
            'n_mfcc': self.n_mfcc,
            'max_time_steps': self.max_time_steps,
            'model_type': self.model_type,
            'emotion_classes': self.label_encoder.classes_.tolist()
        }
        
        import json
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Config saved to {save_dir}/config.json")


def main():
    parser = argparse.ArgumentParser(description='Train speech emotion recognition model')
    parser.add_argument('--train_csv', type=str, default='data/train_metadata.csv',
                        help='Path to train metadata CSV')
    parser.add_argument('--val_csv', type=str, default='data/val_metadata.csv',
                        help='Path to validation metadata CSV')
    parser.add_argument('--test_csv', type=str, default='data/test_metadata.csv',
                        help='Path to test metadata CSV')
    parser.add_argument('--feature_type', type=str, default='mfcc',
                        choices=['mfcc', 'mel', 'combined'],
                        help='Type of features to extract')
    parser.add_argument('--n_mfcc', type=int, default=40,
                        help='Number of MFCC coefficients')
    parser.add_argument('--max_time_steps', type=int, default=174,
                        help='Maximum time steps for padding')
    parser.add_argument('--model_type', type=str, default='bilstm',
                        choices=['rnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm'],
                        help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='Directory to save model and artifacts')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize trainer
    trainer = SERTrainer(
        feature_type=args.feature_type,
        n_mfcc=args.n_mfcc,
        max_time_steps=args.max_time_steps,
        model_type=args.model_type
    )
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        args.train_csv,
        args.val_csv,
        args.test_csv
    )
    
    # Train model
    model_save_path = os.path.join(args.save_dir, f'best_model_{args.model_type}.h5')
    model, history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=model_save_path
    )
    
    # Save artifacts
    trainer.save_artifacts(args.save_dir)
    
    # Save test data for evaluation
    np.save(os.path.join(args.save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.save_dir, 'y_test.npy'), y_test)
    print(f"\nTest data saved to {args.save_dir}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()