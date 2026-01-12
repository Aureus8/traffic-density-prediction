"""
Evaluation Metrics Module

Model performans metrikleri.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMetrics:
    """
    Zaman serisi tahmin modelleri için metrikler.
    
    Metrikler:
    - MAE (Mean Absolute Error)
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² (Coefficient of Determination)
    - SMAPE (Symmetric MAPE)
    """
    
    def __init__(self):
        self.epsilon = 1e-10  # Division by zero önleme
        
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error
        
        Formül: (1/n) * Σ|y_true - y_pred|
        
        Yorumlama: Ortalama mutlak hata. Orijinal ölçekte.
        """
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error
        
        Formül: (1/n) * Σ(y_true - y_pred)²
        
        Yorumlama: Büyük hataları daha fazla cezalandırır.
        """
        return float(np.mean((y_true - y_pred) ** 2))
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error
        
        Formül: √[(1/n) * Σ(y_true - y_pred)²]
        
        Yorumlama: MSE'nin karekökü. Orijinal ölçekte.
        """
        return float(np.sqrt(self.mse(y_true, y_pred)))
    
    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error
        
        Formül: (100/n) * Σ|y_true - y_pred| / |y_true|
        
        Yorumlama: Yüzdesel hata. Hedef < %15.
        
        NOT: y_true = 0 olan değerler problem yaratabilir.
        """
        # y_true = 0 durumunu handle et
        mask = np.abs(y_true) > self.epsilon
        if not mask.any():
            return float('inf')
        
        return float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    
    def smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error
        
        Formül: (100/n) * Σ|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
        
        Yorumlama: MAPE'nin simetrik versiyonu. 0-200 aralığında.
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + self.epsilon
        return float(100 * np.mean(np.abs(y_true - y_pred) / denominator))
    
    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R² (Coefficient of Determination)
        
        Formül: 1 - (SS_res / SS_tot)
        - SS_res = Σ(y_true - y_pred)²
        - SS_tot = Σ(y_true - mean(y_true))²
        
        Yorumlama: 
        - 1.0: Mükemmel tahmin
        - 0.0: Ortalama kadar iyi (baseline)
        - < 0: Ortalamadan kötü
        
        Hedef: > 0.85
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot < self.epsilon:
            return 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Tüm metrikleri hesaplar.
        
        Args:
            y_true: Gerçek değerler
            y_pred: Tahmin edilen değerler
            
        Returns:
            Dict: Tüm metrikler
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        return {
            'mae': round(self.mae(y_true, y_pred), 4),
            'mse': round(self.mse(y_true, y_pred), 4),
            'rmse': round(self.rmse(y_true, y_pred), 4),
            'mape': round(self.mape(y_true, y_pred), 2),
            'smape': round(self.smape(y_true, y_pred), 2),
            'r2': round(self.r2_score(y_true, y_pred), 4)
        }
    
    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Birden fazla modeli karşılaştırır.
        
        Args:
            y_true: Gerçek değerler
            predictions: {'model_name': predictions} dict
            
        Returns:
            Dict: Her model için metrikler
        """
        results = {}
        
        for model_name, y_pred in predictions.items():
            results[model_name] = self.calculate_all(y_true, y_pred)
        
        return results
    
    def check_targets(
        self,
        metrics: Dict[str, float],
        mape_target: float = 15.0,
        r2_target: float = 0.85
    ) -> Dict[str, bool]:
        """
        Hedef metriklere ulaşılıp ulaşılmadığını kontrol eder.
        
        Args:
            metrics: Hesaplanmış metrikler
            mape_target: MAPE hedefi (varsayılan: 15%)
            r2_target: R² hedefi (varsayılan: 0.85)
            
        Returns:
            Dict: Her hedef için pass/fail
        """
        return {
            'mape_target': metrics['mape'] < mape_target,
            'r2_target': metrics['r2'] > r2_target,
            'overall_pass': metrics['mape'] < mape_target and metrics['r2'] > r2_target
        }
    
    def calculate_improvement(
        self,
        baseline_metrics: Dict[str, float],
        model_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Baseline'a göre iyileşmeyi hesaplar.
        
        Args:
            baseline_metrics: Baseline model metrikleri
            model_metrics: Yeni model metrikleri
            
        Returns:
            Dict: Her metrik için % iyileşme
        """
        improvement = {}
        
        for metric in ['mae', 'mse', 'rmse', 'mape', 'smape']:
            if baseline_metrics[metric] > self.epsilon:
                imp = (baseline_metrics[metric] - model_metrics[metric]) / baseline_metrics[metric] * 100
                improvement[f'{metric}_improvement_%'] = round(imp, 2)
        
        # R² için pozitif yönde artış iyi
        r2_imp = (model_metrics['r2'] - baseline_metrics['r2']) / (1 - baseline_metrics['r2'] + self.epsilon) * 100
        improvement['r2_improvement_%'] = round(r2_imp, 2)
        
        return improvement


class TimeSeriesCrossValidator:
    """
    Zaman serisi için cross-validation.
    
    NOT: Normal k-fold CV zaman serileri için uygun DEĞİLDİR!
    Veri kronolojik sırayla bölünmelidir (data leakage önleme).
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        
    def split(
        self,
        data: np.ndarray,
        min_train_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Time series split yapar.
        
        Örnek (n_splits=3):
        - Fold 1: Train [0:33%], Test [33%:50%]
        - Fold 2: Train [0:50%], Test [50%:66%]
        - Fold 3: Train [0:66%], Test [66%:100%]
        
        Args:
            data: Veri array
            min_train_size: Minimum eğitim boyutu
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n = len(data)
        
        if min_train_size is None:
            min_train_size = n // (self.n_splits + 1)
        
        fold_size = (n - min_train_size) // self.n_splits
        
        splits = []
        for i in range(self.n_splits):
            train_end = min_train_size + i * fold_size
            test_start = train_end
            test_end = train_end + fold_size
            
            if i == self.n_splits - 1:
                test_end = n
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def cross_validate(
        self,
        model_class,
        data: np.ndarray,
        target: np.ndarray,
        **model_kwargs
    ) -> Dict[str, List[float]]:
        """
        Cross-validation ile model değerlendirmesi.
        
        Args:
            model_class: Model sınıfı
            data: Feature matrix
            target: Target array
            **model_kwargs: Model parametreleri
            
        Returns:
            Dict: Her fold için metrikler
        """
        metrics = ModelMetrics()
        results = {'mape': [], 'rmse': [], 'r2': []}
        
        for fold, (train_idx, test_idx) in enumerate(self.split(data)):
            # Model oluştur
            model = model_class(**model_kwargs)
            
            # Train
            X_train, y_train = data[train_idx], target[train_idx]
            X_test, y_test = data[test_idx], target[test_idx]
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(len(X_test))
                
                fold_metrics = metrics.calculate_all(y_test, y_pred)
                
                results['mape'].append(fold_metrics['mape'])
                results['rmse'].append(fold_metrics['rmse'])
                results['r2'].append(fold_metrics['r2'])
                
                logger.info(f"Fold {fold + 1}: MAPE={fold_metrics['mape']:.2f}%, "
                           f"R²={fold_metrics['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Fold {fold + 1} failed: {e}")
        
        # Ortalama ve std hesapla
        summary = {}
        for metric_name, values in results.items():
            if values:
                summary[f'{metric_name}_mean'] = round(np.mean(values), 4)
                summary[f'{metric_name}_std'] = round(np.std(values), 4)
        
        return summary


if __name__ == "__main__":
    print("\n=== Metrics Tests ===\n")
    
    # Test data
    np.random.seed(42)
    y_true = np.array([100, 120, 80, 90, 110, 95, 105])
    y_pred = np.array([102, 118, 82, 88, 112, 93, 108])
    
    metrics = ModelMetrics()
    
    # Tek metrik
    print(f"MAE: {metrics.mae(y_true, y_pred):.4f}")
    print(f"RMSE: {metrics.rmse(y_true, y_pred):.4f}")
    print(f"MAPE: {metrics.mape(y_true, y_pred):.2f}%")
    print(f"R²: {metrics.r2_score(y_true, y_pred):.4f}")
    
    # Tüm metrikler
    print("\n=== All Metrics ===")
    all_metrics = metrics.calculate_all(y_true, y_pred)
    for name, value in all_metrics.items():
        print(f"  {name}: {value}")
    
    # Hedef kontrolü
    print("\n=== Target Check ===")
    targets = metrics.check_targets(all_metrics)
    for name, passed in targets.items():
        status = "✓" if passed else "✗"
        print(f"  {name}: {status}")
    
    # Model karşılaştırma
    print("\n=== Model Comparison ===")
    predictions = {
        'Naive': y_true + np.random.normal(0, 10, len(y_true)),
        'Good Model': y_pred,
        'Perfect': y_true
    }
    comparison = metrics.compare_models(y_true, predictions)
    for model, met in comparison.items():
        print(f"  {model}: MAPE={met['mape']:.2f}%, R²={met['r2']:.4f}")
