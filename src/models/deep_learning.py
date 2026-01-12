"""
Deep Learning Models Module

LSTM, GRU ve CNN-LSTM modelleri.
Overfitting kontrolü için Early Stopping, Dropout ve Regularization içerir.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Deep learning models will not be available.")


class EarlyStopping:
    """
    Early Stopping mekanizması.
    
    Overfitting'i önlemek için validation loss iyileşmezse eğitimi durdurur.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, restore_best: bool = True):
        """
        Args:
            patience: Kaç epoch beklenecek
            min_delta: Minimum iyileşme miktarı
            restore_best: En iyi modeli geri yükle
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_model_state = None
        self.early_stop = False
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Her epoch sonunda çağrılır.
        
        Args:
            val_loss: Validation loss
            model: PyTorch model
            
        Returns:
            bool: Eğitim durdurulmalı mı?
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_model_state)
                    logger.info("Early stopping triggered. Restored best model.")
                return True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        return False


class LSTMNetwork(nn.Module):
    """
    LSTM Neural Network mimarisi.
    
    Özellikler:
    - Multi-layer LSTM
    - Dropout regularization
    - Batch normalization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        fc_input_size = hidden_size * self.num_directions
        self.fc1 = nn.Linear(fc_input_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Son hidden state'i al
        if self.bidirectional:
            # İki yönün son hidden state'lerini birleştir
            h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_n = h_n[-1, :, :]
        
        # Dropout
        out = self.dropout(h_n)
        
        # Fully connected
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class GRUNetwork(nn.Module):
    """
    GRU Neural Network mimarisi.
    
    LSTM'den daha hafif, daha hızlı eğitim.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        
        gru_out, h_n = self.gru(x)
        
        # Son hidden state
        out = self.dropout(h_n[-1, :, :])
        out = self.fc(out)
        
        return out


class DeepLearningModel:
    """
    Deep Learning model wrapper.
    
    LSTM ve GRU modellerini eğitir ve tahmin yapar.
    Overfitting kontrolü için:
    - Early Stopping
    - Dropout
    - L2 Regularization (weight decay)
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model_type: str = 'lstm',
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,  # L2 regularization
        device: str = None
    ):
        """
        Args:
            model_type: 'lstm' veya 'gru'
            input_size: Giriş feature sayısı
            hidden_size: Hidden layer boyutu
            num_layers: RNN layer sayısı
            output_size: Çıkış boyutu
            dropout: Dropout oranı (overfitting kontrolü)
            learning_rate: Öğrenme oranı
            weight_decay: L2 regularization (overfitting kontrolü)
            device: 'cuda' veya 'cpu'
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Optimizer with L2 regularization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.is_fitted = False
        
    def _create_model(self) -> nn.Module:
        """Model oluşturur."""
        if self.model_type == 'lstm':
            return LSTMNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.output_size,
                dropout=self.dropout
            )
        elif self.model_type == 'gru':
            return GRUNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.output_size,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32
    ) -> DataLoader:
        """DataLoader oluşturur."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Modeli eğitir.
        
        Args:
            X_train: Eğitim girdileri (n_samples, seq_len, n_features)
            y_train: Eğitim hedefleri (n_samples, output_size)
            X_val: Validation girdileri
            y_val: Validation hedefleri
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            early_stopping_patience: Early stopping sabrı
            verbose: Eğitim çıktısı
            
        Returns:
            Dict: Training history (train_loss, val_loss)
        """
        train_loader = self._prepare_data(X_train, y_train, batch_size)
        
        # Validation loader
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._prepare_data(X_val, y_val, batch_size)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        logger.info(f"Starting training on {self.device}...")
        logger.info(f"Model: {self.model_type.upper()}, "
                   f"Hidden: {self.hidden_size}, "
                   f"Layers: {self.num_layers}, "
                   f"Dropout: {self.dropout}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping (exploding gradients önleme)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                    # Overfitting indicator
                    if train_loss < val_loss * 0.8:
                        msg += " ⚠️ Possible overfitting"
                logger.info(msg)
        
        self.is_fitted = True
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Tahmin yapar.
        
        Args:
            X: Girdi verisi (n_samples, seq_len, n_features)
            
        Returns:
            np.ndarray: Tahminler
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save(self, filepath: str):
        """Modeli kaydeder."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Modeli yükler."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_name(self) -> str:
        return f"{self.model_type.upper()}(h={self.hidden_size},l={self.num_layers})"
    
    def analyze_overfitting(self) -> Dict[str, any]:
        """
        Overfitting analizi yapar.
        
        Returns:
            Dict: Overfitting metrics
        """
        if not self.train_losses or not self.val_losses:
            return {'status': 'No training history available'}
        
        train_final = self.train_losses[-1]
        val_final = self.val_losses[-1]
        
        # Gap analizi
        gap = val_final - train_final
        gap_ratio = val_final / train_final if train_final > 0 else float('inf')
        
        # Min val loss vs final comparison
        min_val_loss = min(self.val_losses)
        min_val_epoch = self.val_losses.index(min_val_loss)
        
        analysis = {
            'train_loss_final': train_final,
            'val_loss_final': val_final,
            'generalization_gap': gap,
            'gap_ratio': gap_ratio,
            'best_val_epoch': min_val_epoch + 1,
            'best_val_loss': min_val_loss,
            'total_epochs': len(self.train_losses)
        }
        
        # Overfitting durumu
        if gap_ratio > 1.5:
            analysis['status'] = 'SEVERE OVERFITTING'
            analysis['recommendation'] = 'Increase dropout, add regularization, or get more data'
        elif gap_ratio > 1.2:
            analysis['status'] = 'MODERATE OVERFITTING'
            analysis['recommendation'] = 'Consider early stopping or reducing model complexity'
        elif gap_ratio < 0.9:
            analysis['status'] = 'POSSIBLE UNDERFITTING'
            analysis['recommendation'] = 'Increase model capacity or train longer'
        else:
            analysis['status'] = 'GOOD FIT'
            analysis['recommendation'] = 'Model is well-balanced'
        
        return analysis


# Wrapper classes for compatibility
class LSTMModel(DeepLearningModel):
    """LSTM model wrapper."""
    
    def __init__(self, **kwargs):
        kwargs['model_type'] = 'lstm'
        super().__init__(**kwargs)


class GRUModel(DeepLearningModel):
    """GRU model wrapper."""
    
    def __init__(self, **kwargs):
        kwargs['model_type'] = 'gru'
        super().__init__(**kwargs)


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping deep learning tests.")
    else:
        print("\n=== Deep Learning Model Tests ===\n")
        
        # Sentetik veri
        np.random.seed(42)
        seq_len = 24
        n_features = 1
        n_samples = 1000
        
        # Sinusoidal veri
        t = np.linspace(0, 100, n_samples + seq_len + 1)
        data = np.sin(t) + 0.1 * np.random.randn(len(t))
        
        # Sequences oluştur
        X = np.array([data[i:i+seq_len] for i in range(n_samples)])
        y = np.array([data[i+seq_len] for i in range(n_samples)])
        
        X = X.reshape(-1, seq_len, n_features)
        y = y.reshape(-1, 1)
        
        # Train/val split
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # LSTM test
        print("Testing LSTM...")
        lstm = LSTMModel(
            input_size=n_features,
            hidden_size=32,
            num_layers=2,
            dropout=0.2
        )
        
        history = lstm.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=50,
            batch_size=32,
            verbose=True
        )
        
        # Prediction
        preds = lstm.predict(X_val[:5])
        print(f"\nLSTM predictions: {preds.flatten()}")
        print(f"Actual values: {y_val[:5].flatten()}")
        
        # Overfitting analizi
        print("\n=== Overfitting Analysis ===")
        analysis = lstm.analyze_overfitting()
        for key, value in analysis.items():
            print(f"  {key}: {value}")
