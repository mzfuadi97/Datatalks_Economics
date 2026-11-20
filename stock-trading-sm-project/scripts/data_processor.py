import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.impute import SimpleImputer
import ta
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and engineer features for ML models"""

    def __init__(self):
        """Initialize feature engineering processors"""

        # Price scalers
        self.price_scalers = {}
        self.vol_scalers = {}
        self.momentum_scalers = {}

        # For model-specific preprocessing
        self.feature_scaler = StandardScaler(with_mean=False)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load preprocessed data

        Args:
            file_path: Path to the data file

        Returns:
            DataFrame with all features
        """
        logger.info(f"Loading data from {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_parquet(file_path)

        logger.info(f"Loaded data with shape: {df.shape}")
        logger.info(f"Dataset columns: {len(df.columns)}")

        # Ensure correct date index
        if df.index.tz_localize:
            df.index = df.index.tz_localize(None)

        return df

    def prepare_target_variable(
        self,
        df: pd.DataFrame,
        target_horizon: int = 5,
        upsample_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Prepare target variable (future returns)

        Args:
            df: DataFrame with price data
            target_horizon: Number of days ahead to predict
            upsample_threshold: Minimum return percentage to label as 'UP'

        Returns:
            DataFrame with target variable
        """
        logger.info(f"Preparing target variable for horizon: {target_horizon} days")

        df = df.copy()
        df['future_returns'] = df['Close'].pct_change(target_horizon).shift(-target_horizon)

        # Create labels
        df['target'] = np.where(
            df['future_returns'] > upsample_threshold,
            1,  # Up
            np.where(
                df['future_returns'] < -upsample_threshold,
                -1,  # Down
                0   # Hold
            )
        )

        # Also keep continuous target for regression
        df['continuous_target'] = df['future_returns']

        logger.info("Target distribution:")
        logger.info(df['target'].value_counts(normalize=True))
        logger.info(f"Class imbalance ratio: {df['target'].value_counts()[1] / df['target'].value_counts()[-1]:.4f}")

        return df

    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment-based features (simplified placeholders)
        In a full implementation, this would integrate:
        - Social media sentiment (Twitter, Reddit)
        - News sentiment analysis
        - Analyst ratings
        - Financial news sentiment

        Args:
            df: DataFrame
            Returns:
                DataFrame with sentiment features
        """
        logger.info("Adding sentiment features")

        # Generate synthetic sentiment based on recent price movement
        df['Simple_Sentiment_Volatility'] = (
            df['returns'].rolling(10).std() * df['returns'].rolling(20).mean()
        )

        # Generate synthetic news sentiment based on returns
        df['News_Sentiment_Score'] = np.where(
            df['returns'] > 0,
            df['returns'] * np.random.uniform(0.8, 1.2, size=len(df)),
            -df['returns'] * np.random.uniform(0.8, 1.2, size=len(df))
        )

        # Analyst target prices placeholder
        df['Analyst_Target_Pct_Bias'] = df['Close'].pct_change(20) * np.random.uniform(0.7, 1.3, size=len(df))

        # Analyst sentiment score based on recent returns
        df['Analyst_Sentiment_Score'] = (
            df['returns'].rolling(5).mean() *
            df['returns'].rolling(20).std() *
            np.random.uniform(0.5, 1.5, size=len(df))
        )

        logger.info("Sentiment features added")

        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        max_lag: int = 5
    ) -> pd.DataFrame:
        """
        Create lagged features for time series context

        Args:
            df: DataFrame
            feature_cols: List of features to lag
            max_lag: Maximum lag period

        Returns:
            DataFrame with lagged features
        """
        logger.info(f"Creating lag features for {len(feature_cols)} features with max lag {max_lag}")

        df = df.copy()
        lagged_dfs = [df]

        for lag in range(1, max_lag + 1):
            lag_df = df[feature_cols].shift(lag).add_suffix(f'_lag_{lag}')
            df = pd.concat([df, lag_df], axis=1)
            lagged_dfs.append(lag_df)

        lagged_cols = [col for lag in range(1, max_lag + 1) for col in df.columns if f'_lag_{lag}' in col]

        logger.info(f"Created {len(lagged_cols)} lagged features")

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        windows: List[int] = [3, 7, 14, 21, 30]
    ) -> pd.DataFrame:
        """
        Create rolling window features

        Args:
            df: DataFrame
            feature_cols: List of features to create windows for
            windows: List of window sizes

        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling features for {len(feature_cols)} features")

        df = df.copy()

        # Create rolling statistics for key features
        key_features = ['returns', 'Volume', 'RSI', 'MACD', 'ADX', 'BB_width']

        for window in windows:
            for feature in key_features:
                if feature in df.columns:
                    df[f'{feature}_roll_mean_{window}'] = df[feature].rolling(window).mean()
                    df[f'{feature}_roll_std_{window}'] = df[feature].rolling(window).std()
                    df[f'{feature}_roll_min_{window}'] = df[feature].rolling(window).min()
                    df[f'{feature}_roll_max_{window}'] = df[feature].rolling(window).max()

        # Normalized price features
        if 'Close' in df.columns:
            for window in windows:
                df['Close_norm_roll_{window}'] = (
                    df['Close'].rolling(window).mean() / df['Close']
                )
                df['Close_roll_adr_{window}'] = (
                    (df['Close'].rolling(window).max() - df['Close'].rolling(window).min()) /
                    df['Close'].rolling(window).mean()
                )

        # Volume ratio rolling features
        if 'VOLUME_RATIO' in df.columns:
            for window in windows:
                df['Volume_ratio_roll_mean_{window}'] = df['VOLUME_RATIO'].rolling(window).mean()
                df['Volume_ratio_roll_std_{window}'] = df['VOLUME_RATIO'].rolling(window).std()

        logger.info("Rolling features created")

        return df

    def create_future_comparison_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create features that compare different timeframes

        Args:
            df: DataFrame
            feature_cols: List of features

        Returns:
            DataFrame with comparison features
        """
        logger.info("Creating future comparison features")

        df = df.copy()

        # Multi-timeframe indicators
        for feature in feature_cols:
            if feature in df.columns:
                df[f'{feature}_5d_shift'] = df[feature].shift(5)
                df[f'{feature}_10d_shift'] = df[feature].shift(10)
                df[f'{feature}_20d_shift'] = df[feature].shift(20)

        # Price momentum across timeframes
        if 'Close' in df.columns:
            df['MA_5_cross_MA_20'] = (
                df['MA_5'] - df['MA_20']
            ) / df['MA_20']
            df['MA_10_cross_MA_50'] = (
                df['MA_10'] - df['MA_50']
            ) / df['MA_50']

        logger.info("Future comparison features created")

        return df

    def add_scaling_features(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str] = None,
        percentile: Tuple[float, float] = (1, 99)
    ) -> pd.DataFrame:
        """
        Add percentile-based scaling features

        Args:
            df: DataFrame
            numeric_cols: List of numeric columns to scale
            percentile: Percentiles to use for scaling

        Returns:
            DataFrame with scaling features
        """
        logger.info("Adding percentile-based scaling features")

        df = df.copy()

        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if col not in ['symbol', 'target', 'continuous_target']:
                try:
                    upper = df[col].quantile(percentile[1])
                    lower = df[col].quantile(percentile[0])

                    # Create robust z-score
                    df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()

                    # Create percentile rank (0-100)
                    df[f'{col}_percentile'] = (df[col].rank() / len(df) * 100).clip(1, 99)

                    # Create boolean features for extreme values
                    df[f'{col}_is_high_{str(percentile[1])}'] = (df[col] > upper).astype(int)
                    df[f'{col}_is_low_{str(percentile[0])}'] = (df[col] < lower).astype(int)

                except Exception as e:
                    logger.warning(f"Could not create scaling features for {col}: {str(e)}")

        logger.info("Scaling features added")

        return df

    def create_engineering_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Create engineered features combining multiple indicators

        Args:
            df: DataFrame
            feature_cols: List of features to use

        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating engineering features")

        df = df.copy()

        # Trend strength indicator
        if all(col in df.columns for col in ['ADX', 'MACD', 'returns']):
            df['Trend_Strength'] = df['ADX'] * df['returns'].abs()
            df['Momentum_Scan'] = df['MACD'] * df['returns'].abs()

        # Volatility regimes with volume
        if 'ROLLING_VOL_20' in df.columns and 'VOLUME_RATIO' in df.columns:
            df['Vol_regime'] = (
                df['ROLLING_VOL_20'] /
                df['Volume'].rolling(50).mean()
            ) * df['VOLUME_RATIO']

        # Oscillator convergence (RSI relative to MACD)
        if 'RSI' in df.columns and 'MACD' in df.columns:
            df['Oscillator_Spread'] = df['RSI'] * df['MACD']

        # Volume-weighted technicals
        if all(col in df.columns for col in ['RSI', 'Volume', 'VOLUME_RATIO']):
            df['Volume_Weighted_RSI'] = df['RSI'] * df['VOLUME_RATIO']

        # Bollinger Band position relative to price
        if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'Close']):
            df['BB_Position'] = (
                df['Close'] - df['BB_lower']
            ) / (df['BB_upper'] - df['BB_lower'])

        # KST confirmation
        if all(col in df.columns for col in ['KST', 'KST_signal']):
            df['KST_Confirmation'] = df['KST'] - df['KST_signal']

        # Aroon crossover
        if all(col in df.columns for col in ['Aroon_Up', 'Aroon_Down']):
            df['Aroon_Crossover'] = df['Aroon_Up'] - df['Aroon_Down']

        # DMI confirmation
        if all(col in df.columns for col in ['DMI_plus', 'DMI_minus']):
            df['DMI_Confidence'] = df['DMI_plus'] / (df['DMI_plus'] + df['DMI_minus'])

        # CMF + OBV trend following indicator
        if all(col in df.columns for col in ['CMF', 'OBV']):
            df['Flow_Trend'] = df['CMF'] * df['OBV']

        # COMBO momentum (RSI + Kiểm nhị giác)
        if all(col in df.columns for col in ['RSI', 'MACD_diff']):
            df['Combo_Momentum'] = df['RSI'] + df['MACD_diff']

        # High-Low compression indicator
        if all(col in df.columns for col in ['High', 'Low', 'Returns']):
            span = 14
            df['HL_Compression'] = (
                df['High'].rolling(span).min() - df['Low'].rolling(span).max()
            ) / df['Close'].rolling(span).mean()

        # VWAP-like feature using volume
        if all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
            df['VWAP_like'] = (
                resample_vwap(df['High'], df['Low'], df['Close'], df['Volume'], freq='1D')
            )

        logger.info("Engineering features created")

        return df

    def get_final_feature_list(self) -> List[str]:
        """
        Get final comprehensive feature list

        Returns:
            List of all features
        """
        return [
            # Price features
            'Open', 'High', 'Low', 'Close', 'Volume',

            # Momentum features
            'returns', 'log_returns',

            # Moving averages
            'MA_7', 'MA_14', 'MA_21', 'MA_30', 'MA_50', 'MA_200',
            f'RSI_7', f'RSI_14', f'RSI_21', f'RSI_30', f'RSI_50',
            f'MA_ratio_7', f'MA_ratio_14', f'MA_ratio_21', f'MA_ratio_30', f'MA_ratio_50',

            # Volatility features
            'ROLLING_VOL_20', 'ROLLING_VOL_50', 'BOLLINGER_BAND', 'ATR',

            # Volume features
            'VOLUME_MA_20', 'VOLUME_MA_50', 'VOLUME_DIFF', 'VOLUME_RATIO',

            # Advanced indicators
            'MACD', 'MACD_signal', 'MACD_diff',
            'KST', 'KST_signal',

            # Oscillators
            'RSI', 'Williams_R', 'Stoch_K', 'Stoch_D',

            # Trend indicators
            'DMI_plus', 'DMI_minus', 'ADX',
            'Aroon_Up', 'Aroon_Down',

            # Volume oscillators
            'CMF', 'OBV', 'Force_Index',

            # Futures comparison
            'future_returns', 'continuous_target', 'target',

            # Market factors
            'BENCHMARK_Close', 'BENCHMARK_returns', 'BENCHMARK_log_returns',
            'MARKET_beta', 'CROSS_RETURN',

            # Macro factors
            'MACRO_NYSA_20_day_avg', 'MACRO_USD_INDEX_signal',

            # Sentiment features
            'Simple_Sentiment_Volatility', 'News_Sentiment_Score',
            'Analyst_Target_Pct_Bias', 'Analyst_Sentiment_Score',

            # Lagged features (example)
            f'Close_lag_1', f'Close_lag_2', f'Close_lag_3', f'Close_lag_4', f'Close_lag_5',
            f'Volume_lag_1', f'Volume_lag_2', f'Volume_lag_3', f'Volume_lag_4', f'Volume_lag_5',

            # Rolling statistics
            'returns_roll_mean_3', 'returns_roll_std_3', 'returns_roll_mean_7', 'returns_roll_std_7',
            'Volume_roll_mean_3', 'Volume_roll_std_3', 'Volume_roll_mean_7', 'Volume_roll_std_7',

            # Advanced engineering features
            'Trend_Strength', 'Momentum_Scan', 'Vol_regime', 'Oscillator_Spread',
            'Volume_Weighted_RSI', 'BB_Position', 'KST_Confirmation', 'Aroon_Crossover',
            'DMI_Confidence', 'Flow_Trend', 'Combo_Momentum', 'HL_Compression'
        ]

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'forward_fill',
        fill_value: float = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df: DataFrame
            strategy: 'forward_fill', 'backward_fill', 'mean', 'median', or custom function

        Returns:
            DataFrame with no missing values
        """
        logger.info(f"Handling missing values with strategy: {strategy}")

        original_size = len(df)
        df = df.copy()

        missing_before = df.isnull().sum().sum()

        if strategy == 'forward_fill':
            df = df.ffill()
            df = df.bfill()
        elif strategy == 'backward_fill':
            df = df.bfill()
            df = df.ffill()
        elif strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif callable(strategy):
            df = df.apply(strategy)
        else:
            df = df.ffill().bfill()

        if fill_value is not None:
            df = df.fillna(fill_value)

        missing_after = df.isnull().sum().sum()
        removed_count = original_size * missing_before / (missing_before - missing_after)

        logger.info(f"Missing values handled: {missing_before} → {missing_after}")
        logger.info(f"Removed {removed_count:.0f} rows due to missing values")

        return df

    def clean_data(
        self,
        df: pd.DataFrame,
        symbol_col: str = 'symbol',
        date_col: str = 'date',
        min_data_points: int = 50
    ) -> pd.DataFrame:
        """
        Clean and validate the dataset

        Args:
            df: DataFrame
            symbol_col: Name of symbol column
            date_col: Name of date column
            min_data_points: Minimum data points required for processing

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning dataset")

        df = df.copy()

        # Remove duplicate rows
        df = df.drop_duplicates()
        original_rows = len(df)
        df = df.drop_duplicates(subset=[symbol_col] + [date_col] if date_col else [symbol_col])
        duplicates_removed = original_rows - len(df)
        logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Remove symbols with insufficient data
        if symbol_col in df.columns:
            symbol_counts = df.groupby(symbol_col).size()
            valid_symbols = symbol_counts[symbol_counts >= min_data_points].index

            filtered_df = df[df[symbol_col].isin(valid_symbols)]
            excluded_symbols = symbol_counts[~symbol_counts.index.isin(valid_symbols)]
            logger.info(f"Removed {len(excluded_symbols)} symbols with insufficient data")

            df = filtered_df

        # Ensure date/index is sorted
        if date_col in df.columns:
            df = df.sort_values([symbol_col, date_col])
            df = df.drop_duplicates(subset=[symbol_col, date_col], keep='first')
        else:
            df = df.sort_index()

        logger.info(f"Cleaned dataset shape: {df.shape}")

        return df


def resample_vwap(high, low, close, volume, freq='1D'):
    """Calculate volume-weighted average price"""
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df.index.name = 'date'

    resampled = (
        df.resample(freq).apply(lambda x: (x['high'] + x['low'] + x['close'] * 2).sum() / (6 * x['volume'].sum()))
    )

    return resampled


if __name__ == "__main__":
    from config import RAW_DIR, PROCESSED_DIR

    processor = DataProcessor()

    raw_path = RAW_DIR / "stock_data.parquet"
    processed_dir = PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Data Processing Pipeline")
    print("=" * 50)

    if not raw_path.exists():
        raise SystemExit(f"Raw data not found: {raw_path}. Run data_collector.py first.")

    df = pd.read_parquet(raw_path)
    print(f"Loaded raw data: {df.shape}")

    # Normalize date column name
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    if 'index' in df.columns and 'date' not in df.columns:
        df = df.rename(columns={'index': 'date'})

    df['date'] = pd.to_datetime(df['date'])

    # Sort by symbol and date, drop duplicate (symbol, date)
    df = df.sort_values(['symbol', 'date'])
    df = df.drop_duplicates(subset=['symbol', 'date'], keep='first')

    # Create target variable per symbol
    processed_frames = []
    for symbol, group in df.groupby('symbol'):
        group = group.copy().sort_values('date')
        group = processor.prepare_target_variable(group, target_horizon=5, upsample_threshold=0.02)
        processed_frames.append(group)

    df = pd.concat(processed_frames, ignore_index=True)

    # Handle missing values (forward/back fill within each symbol)
    df = df.groupby('symbol', group_keys=False).apply(lambda g: g.ffill().bfill())
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['target'])

    # Save processed dataset
    output_path = processed_dir / "processed_data.parquet"
    df.to_parquet(output_path)
    print(f"Saved processed data: {output_path}")
    print(f"Processed shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
