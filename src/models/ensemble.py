"""
Ensemble Models Module

Birden fazla modeli kombine eden ensemble yöntemleri.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble Model
    
    Birden fazla modelin tahminlerini kombine eder.
    
    Stratejiler:
    - simple_average: Basit ortalama
    - weighted_average: Ağırlıklı ortalama (performansa göre)
    - stacking: Meta-learner ile kombinasyon
    """
    
    def __init__(
        self,
        models: List[Any],
        strategy: str = 'weighted_average',
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            models: Model listesi (fit() ve predict() metodları olmalı)
            strategy: 'simple_average', 'weighted_average', 'stacking'
            weights: Manuel ağırlıklar (weighted_average için)
        """
        self.models = models
        self.strategy = strategy
        self.weights = weights
        self.is_fitted = False
        self.model_performances = {}
        
    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_col: str = 'density',
        **kwargs
    ) -> 'EnsembleModel':
        """
        Tüm modelleri eğitir ve ağırlıkları hesaplar.
        
        Args:
            train_data: Eğitim verisi
            val_data: Validation verisi (ağırlık hesabı için)
            target_col: Hedef sütun
            **kwargs: Model-specific parametreler
        """
        logger.info(f"Training ensemble with {len(self.models)} models...")
        
        y_val = val_data[target_col].values
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                # Model eğit
                model.fit(train_data, target_col=target_col, **kwargs)
                
                # Validation tahmini
                n_val = len(val_data)
                pred = model.predict(n_val)
                predictions.append(pred)
                
                # Performans hesapla (MAPE)
                mape = np.mean(np.abs((y_val - pred) / (y_val + 1e-10))) * 100
                self.model_performances[model.get_name()] = {
                    'mape': mape,
                    'index': i
                }
                
                logger.info(f"  {model.get_name()}: MAPE = {mape:.2f}%")
                
            except Exception as e:
                logger.error(f"Failed to train {model.get_name()}: {e}")
                continue
        
        # Ağırlıkları hesapla
        if self.strategy == 'weighted_average' and self.weights is None:
            self._calculate_weights()
        
        self.is_fitted = True
        logger.info(f"Ensemble training completed. Strategy: {self.strategy}")
        
        return self
    
    def _calculate_weights(self):
        """
        Model performanslarına göre ağırlıkları hesaplar.
        
        Düşük MAPE = Yüksek ağırlık (inverse weighting).
        """
        mapes = [perf['mape'] for perf in self.model_performances.values()]
        
        # Inverse MAPE (daha düşük hata = daha yüksek ağırlık)
        inv_mapes = [1 / (mape + 1e-10) for mape in mapes]
        total = sum(inv_mapes)
        
        self.weights = [w / total for w in inv_mapes]
        
        logger.info(f"Calculated weights: {self.weights}")
    
    def predict(self, n_steps: int = 1, **kwargs) -> np.ndarray:
        """
        Ensemble tahmini yapar.
        
        Args:
            n_steps: Tahmin adım sayısı
            **kwargs: Model-specific parametreler
            
        Returns:
            np.ndarray: Kombine tahmin
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(n_steps, **kwargs)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {model.get_name()}: {e}")
                continue
        
        predictions = np.array(predictions)
        
        # Strateji uygula
        if self.strategy == 'simple_average':
            return np.mean(predictions, axis=0)
        
        elif self.strategy == 'weighted_average':
            if self.weights is None or len(self.weights) != len(predictions):
                # Fallback to simple average
                return np.mean(predictions, axis=0)
            
            weighted_sum = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                weighted_sum += pred * weight
            return weighted_sum
        
        elif self.strategy == 'median':
            return np.median(predictions, axis=0)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def get_model_contributions(self) -> Dict[str, float]:
        """Her modelin ensemble'a katkısını döndürür."""
        contributions = {}
        
        if self.weights is None:
            equal_weight = 1.0 / len(self.models)
            for model in self.models:
                contributions[model.get_name()] = equal_weight
        else:
            for (name, perf), weight in zip(self.model_performances.items(), self.weights):
                contributions[name] = weight
        
        return contributions
    
    def get_name(self) -> str:
        model_names = [m.get_name() for m in self.models]
        return f"Ensemble({self.strategy})[{', '.join(model_names)}]"


class StackingEnsemble:
    """
    Stacking Ensemble
    
    Meta-learner kullanarak modellerin tahminlerini kombine eder.
    
    Level 0: Base modeller
    Level 1: Meta-learner (base model tahminlerini girdi olarak alır)
    """
    
    def __init__(
        self,
        base_models: List[Any],
        meta_learner: Any = None
    ):
        """
        Args:
            base_models: Base model listesi
            meta_learner: Meta-learner model (varsayılan: Ridge regression)
        """
        self.base_models = base_models
        
        # Varsayılan meta-learner
        if meta_learner is None:
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
        else:
            self.meta_learner = meta_learner
            
        self.is_fitted = False
        
    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_col: str = 'density',
        **kwargs
    ) -> 'StackingEnsemble':
        """
        Stacking ensemble eğitimi.
        
        1. Base modelleri eğit
        2. Base modellerden validation tahminleri al
        3. Meta-learner'ı base tahminler + gerçek değerlerle eğit
        """
        logger.info(f"Training stacking ensemble with {len(self.base_models)} base models...")
        
        y_val = val_data[target_col].values
        n_val = len(val_data)
        
        # Base model tahminleri
        base_predictions = []
        
        for model in self.base_models:
            model.fit(train_data, target_col=target_col, **kwargs)
            pred = model.predict(n_val)
            base_predictions.append(pred)
            logger.info(f"  Trained: {model.get_name()}")
        
        # Meta-learner için girdi hazırla
        X_meta = np.column_stack(base_predictions)
        
        # Meta-learner eğit
        self.meta_learner.fit(X_meta, y_val)
        logger.info("Meta-learner trained")
        
        self.is_fitted = True
        return self
    
    def predict(self, n_steps: int = 1, **kwargs) -> np.ndarray:
        """
        Stacking tahmini.
        
        1. Base modellerden tahmin al
        2. Tahminleri meta-learner'a ver
        3. Final tahmini döndür
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Base tahminler
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(n_steps, **kwargs)
            base_predictions.append(pred)
        
        # Meta-learner tahmini
        X_meta = np.column_stack(base_predictions)
        final_prediction = self.meta_learner.predict(X_meta)
        
        return final_prediction
    
    def get_name(self) -> str:
        base_names = [m.get_name() for m in self.base_models]
        return f"Stacking[{', '.join(base_names)}]"


def create_default_ensemble(include_deep_learning: bool = False) -> EnsembleModel:
    """
    Varsayılan ensemble oluşturur.
    
    Args:
        include_deep_learning: Derin öğrenme modelleri dahil et
        
    Returns:
        EnsembleModel: Hazır ensemble
    """
    from .baseline import NaiveModel, MovingAverageModel, ARIMAModel
    from .statistical import SARIMAXModel
    
    models = [
        MovingAverageModel(window=24),
        ARIMAModel(order=(2, 1, 1)),
        SARIMAXModel(order=(1, 1, 1), seasonal_order=(1, 0, 1, 24))
    ]
    
    if include_deep_learning:
        try:
            from .deep_learning import LSTMModel
            # DL modeller sequence input gerektirir, bu yüzden ayrı işlem gerekir
            logger.warning("Deep learning models require different input format. "
                         "Consider using them separately.")
        except ImportError:
            pass
    
    return EnsembleModel(models, strategy='weighted_average')


if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.models.baseline import NaiveModel, MovingAverageModel, ARIMAModel
    
    print("\n=== Ensemble Model Tests ===\n")
    
    # Veri hazırla
    loader = DataLoader()
    df = loader.generate_synthetic_data(
        num_zones=1,
        start_date="2023-01-01",
        end_date="2023-06-30"
    )
    
    preprocessor = DataPreprocessor()
    train, val, test = preprocessor.time_series_split(df)
    
    # Base modeller
    models = [
        NaiveModel(),
        MovingAverageModel(window=24),
        ARIMAModel(order=(2, 1, 1))
    ]
    
    # Simple ensemble
    print("Testing Simple Average Ensemble...")
    ensemble = EnsembleModel(models, strategy='simple_average')
    ensemble.fit(train, val)
    pred = ensemble.predict(5)
    print(f"Ensemble prediction: {pred}")
    
    # Weighted ensemble
    print("\nTesting Weighted Average Ensemble...")
    models2 = [
        NaiveModel(),
        MovingAverageModel(window=24),
        ARIMAModel(order=(2, 1, 1))
    ]
    weighted_ensemble = EnsembleModel(models2, strategy='weighted_average')
    weighted_ensemble.fit(train, val)
    pred_weighted = weighted_ensemble.predict(5)
    print(f"Weighted prediction: {pred_weighted}")
    print(f"Model contributions: {weighted_ensemble.get_model_contributions()}")
