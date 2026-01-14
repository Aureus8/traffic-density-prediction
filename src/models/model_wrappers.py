"""
Shared Model Wrapper Classes

Contains all model class definitions used by both:
- scripts/train_models.py (for training and saving)
- src/api/main.py (for loading and predicting)

This shared module prevents AttributeError during pickle deserialization.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class NaiveModel:
    """Naive prediction model - uses last observed value."""
    
    def __init__(self):
        self.last_value = None
    
    def fit(self, y: np.ndarray) -> 'NaiveModel':
        """Fit the model - just stores the last value."""
        self.last_value = y[-1] if len(y) > 0 else 0
        return self
    
    def predict(self, n_periods: int = 1, y_prev: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict next n periods."""
        if y_prev is not None and len(y_prev) > 0:
            return np.full(n_periods, y_prev[-1])
        return np.full(n_periods, self.last_value)


class MovingAverageModel:
    """Moving average prediction model."""
    
    def __init__(self, window: int = 24):
        self.window = window
        self.history = None
    
    def fit(self, y: np.ndarray) -> 'MovingAverageModel':
        """Fit the model - stores history for moving average."""
        self.history = np.array(y)[-self.window:]
        return self
    
    def predict(self, n_periods: int = 1, y_prev: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict next n periods."""
        if y_prev is not None:
            history = np.concatenate([self.history, y_prev])[-self.window:]
        else:
            history = self.history
        
        predictions = []
        for _ in range(n_periods):
            pred = np.mean(history[-self.window:])
            predictions.append(pred)
            history = np.append(history, pred)
        
        return np.array(predictions)


class ARIMAModelWrapper:
    """ARIMA model wrapper using statsmodels."""
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 1)):
        self.order = order
        self.model = None
        self.fitted = None
        self._fallback_mean = 0
    
    def fit(self, y: np.ndarray) -> 'ARIMAModelWrapper':
        """Fit the ARIMA model."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            y = np.array(y)[-1000:] if len(y) > 1000 else np.array(y)
            self.model = ARIMA(y, order=self.order)
            self.fitted = self.model.fit()
        except Exception:
            self.fitted = None
            self._fallback_mean = np.mean(y)
        return self
    
    def predict(self, n_periods: int = 1) -> np.ndarray:
        """Predict next n periods."""
        if self.fitted is not None:
            try:
                forecast = self.fitted.forecast(steps=n_periods)
                return np.array(forecast)
            except Exception:
                pass
        return np.full(n_periods, self._fallback_mean)


class SARIMAXModelWrapper:
    """SARIMAX model wrapper with exogenous variables."""
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted = None
        self._fallback_mean = 0
    
    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None) -> 'SARIMAXModelWrapper':
        """Fit the SARIMAX model."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            max_samples = 500
            if len(y) > max_samples:
                y = np.array(y)[-max_samples:]
                if exog is not None:
                    exog = np.array(exog)[-max_samples:]
            
            self.model = SARIMAX(
                y, exog=exog, order=self.order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted = self.model.fit(disp=False, maxiter=50)
        except Exception:
            self.fitted = None
            self._fallback_mean = np.mean(y)
        return self
    
    def predict(self, n_periods: int = 1, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict next n periods."""
        if self.fitted is not None:
            try:
                forecast = self.fitted.forecast(steps=n_periods, exog=exog)
                return np.array(forecast)
            except Exception:
                pass
        return np.full(n_periods, self._fallback_mean)


class SimpleLSTMModel:
    """Simple LSTM model for time series prediction."""
    
    def __init__(
        self,
        sequence_length: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.scaler = None
        self.device = 'cpu'
        self._fallback_mean = 0
        self._last_sequence = None
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def fit(self, y: np.ndarray, epochs: int = 50, batch_size: int = 64,
            early_stopping_patience: int = 10) -> 'SimpleLSTMModel':
        """Fit the LSTM model."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.preprocessing import MinMaxScaler
            
            self.scaler = MinMaxScaler()
            y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
            X, y_seq = self._create_sequences(y_scaled)
            
            if len(X) == 0:
                self._fallback_mean = np.mean(y)
                return self
            
            X_tensor = torch.FloatTensor(X).unsqueeze(-1)
            y_tensor = torch.FloatTensor(y_seq)
            
            split_idx = int(len(X_tensor) * 0.8)
            X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
            
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            class LSTMNet(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                        batch_first=True, dropout=dropout)
                    self.fc = nn.Linear(hidden_size, 1)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :]).squeeze()
            
            self.model = LSTMNet(1, self.hidden_size, self.num_layers, self.dropout)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                self.model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()
                
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = criterion(val_pred, y_val).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break
            
            self._last_sequence = y_scaled[-self.sequence_length:]
            
        except Exception:
            self._fallback_mean = np.mean(y)
        return self
    
    def predict(self, n_periods: int = 1) -> np.ndarray:
        """Predict next n periods."""
        if self.model is None:
            return np.full(n_periods, self._fallback_mean)
        
        try:
            import torch
            self.model.eval()
            predictions = []
            sequence = self._last_sequence.copy()
            
            with torch.no_grad():
                for _ in range(n_periods):
                    X = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)
                    pred = self.model(X).item()
                    predictions.append(pred)
                    sequence = np.append(sequence[1:], pred)
            
            predictions = self.scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).flatten()
            return predictions
        except Exception:
            return np.full(n_periods, self._fallback_mean)


class EnsembleModel:
    """Ensemble model combining multiple base models."""
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {}
    
    def calculate_weights_from_metrics(self, metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate ensemble weights based on inverse MAPE."""
        weights = {}
        total_inv_mape = 0
        
        for name, model_metrics in metrics.items():
            if name in self.models and model_metrics.get('mape', 100) > 0:
                inv_mape = 1 / model_metrics['mape']
                weights[name] = inv_mape
                total_inv_mape += inv_mape
        
        if total_inv_mape > 0:
            weights = {k: v / total_inv_mape for k, v in weights.items()}
        
        self.weights = weights
        return weights
    
    def predict(self, n_periods: int = 1, **kwargs) -> np.ndarray:
        """Predict using weighted ensemble."""
        predictions = []
        total_weight = 0
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0 / len(self.models))
            try:
                pred = model.predict(n_periods)
                predictions.append(pred * weight)
                total_weight += weight
            except Exception:
                pass
        
        if predictions:
            return np.sum(predictions, axis=0) / total_weight
        return np.zeros(n_periods)
