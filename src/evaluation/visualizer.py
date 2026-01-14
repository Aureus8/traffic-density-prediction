"""
Performance Visualizer Module

Model performans grafikleri.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Visualization will not be available.")

# Seaborn for styling
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class PerformanceVisualizer:
    """
    Model performans görselleştirme.
    
    Grafikler:
    - Tahmin vs Gerçek
    - Residual analizi
    - Learning curves (overfitting detection)
    - Model karşılaştırma
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', figsize: tuple = (12, 6)):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not installed.")
        
        self.figsize = figsize
        
        # Style ayarla
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Predictions vs Actual",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Tahmin vs gerçek değer grafiği.
        
        Args:
            y_true: Gerçek değerler
            y_pred: Tahminler
            dates: Tarih indeksi
            title: Grafik başlığı
            save_path: Kayıt yolu
            
        Returns:
            Figure: matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        x = dates if dates is not None else np.arange(len(y_true))
        
        # Üst grafik: Tahmin vs Gerçek
        axes[0].plot(x, y_true, label='Actual', color='#2E86AB', linewidth=2)
        axes[0].plot(x, y_pred, label='Predicted', color='#A23B72', 
                     linewidth=2, linestyle='--')
        axes[0].fill_between(x, y_true, y_pred, alpha=0.2, color='gray')
        axes[0].set_ylabel('Density')
        axes[0].set_title(title)
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Alt grafik: Hata
        error = y_true - y_pred
        axes[1].bar(x, error, color=np.where(error >= 0, '#28A745', '#DC3545'), alpha=0.6)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_ylabel('Error (Actual - Predicted)')
        axes[1].set_xlabel('Time')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_learning_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Learning Curves",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Learning curve grafiği - Overfitting tespiti için kritik!
        
        Yorumlama:
        - Train ↓, Val ↓: İyi öğrenme
        - Train ↓, Val ↑: OVERFITTING
        - Her ikisi de yüksek ve düz: UNDERFITTING
        
        Args:
            train_losses: Eğitim loss değerleri
            val_losses: Validation loss değerleri
            title: Grafik başlığı
            save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, label='Training Loss', 
                color='#2E86AB', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_losses, label='Validation Loss', 
                color='#A23B72', linewidth=2, marker='s', markersize=3)
        
        # Best epoch işaretle
        best_epoch = np.argmin(val_losses) + 1
        best_val = min(val_losses)
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax.scatter([best_epoch], [best_val], color='green', s=100, zorder=5)
        
        # Overfitting analizi
        final_gap = val_losses[-1] - train_losses[-1]
        gap_ratio = val_losses[-1] / train_losses[-1] if train_losses[-1] > 0 else float('inf')
        
        if gap_ratio > 1.5:
            status = "⚠️ OVERFITTING DETECTED"
            color = 'red'
        elif gap_ratio < 0.9 and train_losses[-1] > 0.1:
            status = "⚠️ POSSIBLE UNDERFITTING"
            color = 'orange'
        else:
            status = "✓ Good Fit"
            color = 'green'
        
        ax.text(0.02, 0.98, status, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', color=color, fontweight='bold')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Residual (artık) analizi.
        
        İyi bir modelde residual'lar:
        - Sıfır etrafında dağılmalı
        - Homojen varyans göstermeli
        - Normal dağılıma yakın olmalı
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # 1. Residual vs Fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, color='#2E86AB')
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # 2. Histogram
        axes[0, 1].hist(residuals, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='black', linestyle='--')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residual Distribution')
        
        # 3. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # 4. Residual over time
        axes[1, 1].plot(residuals, color='#28A745', alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].axhline(y=np.std(residuals) * 2, color='orange', linestyle=':', label='+2σ')
        axes[1, 1].axhline(y=-np.std(residuals) * 2, color='orange', linestyle=':', label='-2σ')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].set_title('Residuals Over Time')
        axes[1, 1].legend()
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        metrics_to_plot: List[str] = ['mape', 'rmse', 'r2'],
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Model karşılaştırma grafiği.
        
        Args:
            comparison_results: {model_name: {metric: value}} format
            metrics_to_plot: Gösterilecek metrikler
            title: Grafik başlığı
            save_path: Kayıt yolu
        """
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(self.figsize[0] * 1.5, self.figsize[1]))
        
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(comparison_results.keys())
        colors = plt.cm.husl(np.linspace(0, 0.8, len(model_names)))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [comparison_results[m].get(metric, 0) for m in model_names]
            
            bars = axes[i].bar(model_names, values, color=colors)
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'{metric.upper()}')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Değerleri bar üstüne yaz
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[i].annotate(f'{val:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
            
            # R² için threshold çizgisi
            if metric == 'r2':
                axes[i].axhline(y=0.85, color='green', linestyle='--', 
                               label='Target (0.85)', alpha=0.7)
                axes[i].legend()
            # MAPE için threshold
            elif metric == 'mape':
                axes[i].axhline(y=15, color='green', linestyle='--', 
                               label='Target (15%)', alpha=0.7)
                axes[i].legend()
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_forecast(
        self,
        historical: np.ndarray,
        forecast: np.ndarray,
        confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        historical_dates: Optional[pd.DatetimeIndex] = None,
        forecast_dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Forecast",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Tahmin görselleştirmesi.
        
        Args:
            historical: Geçmiş değerler
            forecast: Tahminler
            confidence_interval: (lower, upper) güven aralığı
            historical_dates: Geçmiş tarihler
            forecast_dates: Tahmin tarihleri
            title: Grafik başlığı
            save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_hist = len(historical)
        n_fore = len(forecast)
        
        if historical_dates is not None and forecast_dates is not None:
            x_hist = historical_dates
            x_fore = forecast_dates
        else:
            x_hist = np.arange(n_hist)
            x_fore = np.arange(n_hist, n_hist + n_fore)
        
        # Geçmiş
        ax.plot(x_hist, historical, label='Historical', color='#2E86AB', linewidth=2)
        
        # Tahmin
        ax.plot(x_fore, forecast, label='Forecast', color='#A23B72', 
                linewidth=2, linestyle='--')
        
        # Güven aralığı
        if confidence_interval is not None:
            lower, upper = confidence_interval
            ax.fill_between(x_fore, lower, upper, alpha=0.2, color='#A23B72',
                          label='Confidence Interval')
        
        # Dikey çizgi (tahmin başlangıcı)
        ax.axvline(x=x_hist[-1], color='gray', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Skipping visualization tests.")
    else:
        print("\n=== Visualizer Tests ===\n")
        
        # Test data
        np.random.seed(42)
        n = 100
        y_true = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 5, n)
        y_pred = y_true + np.random.normal(0, 8, n)
        
        # Learning curves
        train_losses = [0.5 / (i + 1) + 0.1 + np.random.uniform(0, 0.02) for i in range(50)]
        val_losses = [0.5 / (i + 1) + 0.15 + np.random.uniform(0, 0.03) for i in range(50)]
        
        viz = PerformanceVisualizer()
        
        # Prediction plot
        fig1 = viz.plot_predictions(y_true[:50], y_pred[:50], title="Test Predictions")
        plt.close(fig1)
        
        # Learning curves
        fig2 = viz.plot_learning_curves(train_losses, val_losses, title="Training Progress")
        plt.close(fig2)
        
        # Model comparison
        comparison = {
            'Naive': {'mape': 25.5, 'rmse': 15.2, 'r2': 0.72},
            'ARIMA': {'mape': 18.3, 'rmse': 12.1, 'r2': 0.81},
            'LSTM': {'mape': 12.1, 'rmse': 8.5, 'r2': 0.89}
        }
        fig3 = viz.plot_model_comparison(comparison)
        plt.close(fig3)
        
        print("Visualization tests completed successfully!")
