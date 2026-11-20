import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import ta
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collect and fetch stock market data from multiple sources"""

    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        """
        Initialize data collector

        Args:
            symbols: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(
        self,
        interval: str = '1d',
        features: List[str] = None,
        download_all: bool = False
    ) -> pd.DataFrame:
        """
        Fetch data for all symbols
        """
        all_data = []

        logger.info(f"Fetching data for {len(self.symbols)} symbols from {self.start_date} to {self.end_date}")

        for symbol in self.symbols:
            try:
                logger.info(f"Fetching data for {symbol}")

                # Fetch basic OHLCV data
                df = yf.download(
                    symbol,
                    start=self.start_date.strftime('%Y-%m-%d'),
                    end=self.end_date.strftime('%Y-%m-%d'),
                    interval=interval
                )
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # yf.download returns a MultiIndex if multiple tickers, but we pass one.
                # However, sometimes it returns a MultiIndex anyway. Need to flatten it.
                if isinstance(df.columns, pd.MultiIndex):
                     df.columns = df.columns.get_level_values(0)

                df['symbol'] = symbol
                all_data.append(df)

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No data could be fetched from any symbol")

        combined_df = pd.concat(all_data)
        combined_df.reset_index(inplace=True)

        logger.info(f"Combined dataset shape: {combined_df.shape}")
        
        return combined_df


    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for feature engineering

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with calculated indicators
        """
        logger.info("Calculating technical indicators")

        result_df = df.copy()

        # Price-based indicators
        result_df['returns'] = result_df['Close'].pct_change()
        result_df['log_returns'] = np.log(result_df['Close'] / result_df['Close'].shift(1))

        # Moving averages
        for window in [7, 14, 21, 30, 50, 200]:
            sma_indicator = ta.trend.SMAIndicator(
                close=result_df['Close'],
                window=window
            )
            result_df[f'MA_{window}'] = sma_indicator.sma_indicator()

            ema_indicator = ta.trend.EMAIndicator(
                close=result_df['Close'],
                window=window
            )
            result_df[f'EMA_{window}'] = ema_indicator.ema_indicator()

            rsi_indicator = ta.momentum.RSIIndicator(
                close=result_df['Close'],
                window=window
            )
            result_df[f'RSI_{window}'] = rsi_indicator.rsi()

            result_df[f'MA_ratio_{window}'] = result_df[f'Close'] / result_df[f'MA_{window}']

        # Volatility indicators
        result_df['ROLLING_VOL_20'] = result_df['returns'].rolling(20).std()
        result_df['ROLLING_VOL_50'] = result_df['returns'].rolling(50).std()

        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(
            close=result_df['Close'],
            window=20,
            window_dev=2
        )
        result_df['BB_HIGH'] = bb_indicator.bollinger_hband()
        result_df['BB_LOW'] = bb_indicator.bollinger_lband()
        result_df['BB_MID'] = bb_indicator.bollinger_mavg()
        result_df['BOLLINGER_BAND'] = (result_df['Close'] - result_df['BB_MID']) / (result_df['BB_HIGH'] - result_df['BB_LOW'])



        # Momentum indicators
        for period in [1, 2, 3, 5, 10, 20]:
            roc_indicator = ta.momentum.ROCIndicator(
                close=result_df['Close'],
                window=period
            )
            result_df[f'ROC_{period}'] = roc_indicator.roc()


        # ATR for volatility measurement
        atr_indicator = ta.volatility.AverageTrueRange(
            high=result_df['High'],
            low=result_df['Low'],
            close=result_df['Close'],
            window=14
        )
        result_df['ATR'] = atr_indicator.average_true_range()



        # Volume indicators
        # Calculate moving average for volume comparison
        sma_vol_20 = ta.trend.SMAIndicator(
            close=result_df['Volume'],
            window=20
        )
        sma_vol_50 = ta.trend.SMAIndicator(
            close=result_df['Volume'],
            window=50
        )
        result_df['VOLUME_MA_20'] = sma_vol_20.sma_indicator()
        result_df['VOLUME_MA_50'] = sma_vol_50.sma_indicator()
        result_df['VOLUME_DIFF'] = result_df['Volume'].values - result_df['VOLUME_MA_20'].values
        result_df['VOLUME_RATIO'] = result_df['Volume'] / result_df['VOLUME_MA_20']


        # MACD - Many models including momentum
        macd = ta.trend.MACD(
            close=result_df['Close'],
            window_fast=12,
            window_slow=26,
            window_sign=9
        )
        result_df['MACD'] = macd.macd()
        result_df['MACD_signal'] = macd.macd_signal()
        result_df['MACD_diff'] = macd.macd_diff()


        # Oscillators
        rsi = ta.momentum.RSIIndicator(
            close=result_df['Close'],
            window=14
        )
        result_df['RSI'] = rsi.rsi()

        kst = ta.trend.KSTIndicator(
            close=result_df['Close'],
            roc1=10,
            roc2=15,
            roc3=20,
            roc4=30
        )
        result_df['KST'] = kst.kst()
        result_df['KST_signal'] = kst.kst_sig()

        # Chaikin Money Flow (CMF)
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            high=result_df['High'],
            low=result_df['Low'],
            close=result_df['Close'],
            volume=result_df['Volume']
        )
        result_df['CMF'] = cmf.chaikin_money_flow()

        # On-Balance Volume (OBV)
        obv = ta.volume.OnBalanceVolumeIndicator(
            close=result_df['Close'],
            volume=result_df['Volume']
        )
        result_df['OBV'] = obv.on_balance_volume()

        # Force Index
        result_df['Force_Index'] = ta.volume.force_index(
            close=result_df['Close'],
            volume=result_df['Volume']
        )

        # Rate of Change (ROC) from ta library
        roc = ta.momentum.ROCIndicator(
            close=result_df['Close'],
            window=14
        )
        result_df['ROC_14'] = roc.roc()

        # Williams %R
        williams_r = ta.momentum.WilliamsRIndicator(
            high=result_df['High'],
            low=result_df['Low'],
            close=result_df['Close'],
            lbp=14
        )
        result_df['Williams_R'] = williams_r.williams_r()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=result_df['High'],
            low=result_df['Low'],
            close=result_df['Close'],
            window=14,
            smooth_window=3
        )
        result_df['Stoch_K'] = stoch.stoch()
        result_df['Stoch_D'] = stoch.stoch_signal()

        # DMI indicators (using ADXIndicator which provides DMI +/- and ADX)
        adx = ta.trend.ADXIndicator(
            high=result_df['High'],
            low=result_df['Low'],
            close=result_df['Close'],
            window=14
        )
        result_df['DMI_plus'] = adx.adx_pos()
        result_df['DMI_minus'] = adx.adx_neg()
        result_df['ADX'] = adx.adx()

        # Aroon indicators
        aroon = ta.trend.AroonIndicator(
            high=result_df['High'],
            low=result_df['Low'],
            window=25
        )
        result_df['Aroon_Up'] = aroon.aroon_up()
        result_df['Aroon_Down'] = aroon.aroon_down()

        # Bollinger Bands (more detailed)
        bb = ta.volatility.BollingerBands(
            close=result_df['Close'],
            window=20,
            window_dev=2
        )
        result_df['BB_upper'] = bb.bollinger_hband()
        result_df['BB_middle'] = bb.bollinger_mavg()
        result_df['BB_lower'] = bb.bollinger_lband()
        result_df['BB_width'] = bb.bollinger_wband()

        # Purchase Power Parity slope (simplified)
        result_df['PPP_slope'] = np.polyfit(range(len(result_df['Close'])), result_df['Close'], 1)[0]

        logger.info(f"Created {len(result_df.columns)} columns including technical indicators")

        return result_df

    def calculate_market_factors(self, df: pd.DataFrame, benchmark_symbol: str) -> pd.DataFrame:
        """
        Calculate market-wide factors and correlations

        Args:
            df: Price data DataFrame
            benchmark_symbol: S&P 500 ticker symbol

        Returns:
            DataFrame with market factors
        """
        logger.info(f"Calculating market factors vs {benchmark_symbol}")

        try:
            bench = yf.download(
                benchmark_symbol,
                start=self.start_date,
                end=self.end_date,
                interval='1d',
                auto_adjust=True,
                progress=False,
            )

            if bench is None or bench.empty:
                logger.warning(f"No benchmark data found for {benchmark_symbol}")
                return df

            if isinstance(bench.columns, pd.MultiIndex):
                bench.columns = bench.columns.get_level_values(0)

            bench = bench.reset_index()
            date_col = 'Date' if 'Date' in bench.columns else bench.columns[0]
            bench = bench[[date_col, 'Close']].rename(
                columns={date_col: 'Date', 'Close': 'BENCHMARK_Close'}
            )
            bench['BENCHMARK_returns'] = bench['BENCHMARK_Close'].pct_change()
            bench['BENCHMARK_log_returns'] = np.log(
                bench['BENCHMARK_Close'] / bench['BENCHMARK_Close'].shift(1)
            )

            left = df.copy()
            if 'date' in left.columns and 'Date' not in left.columns:
                left = left.rename(columns={'date': 'Date'})
            left['Date'] = pd.to_datetime(left['Date']).dt.tz_localize(None)
            bench['Date'] = pd.to_datetime(bench['Date']).dt.tz_localize(None)

            left = left.merge(bench, on='Date', how='left')
            left['MARKET_beta'] = left['returns'] / left['BENCHMARK_returns'].replace(0, np.nan)
            left['CROSS_RETURN'] = left['returns'] - left['BENCHMARK_returns']
            df = left

        except Exception as e:
            logger.warning(f"Could not calculate market factors: {str(e)}")

        logger.info("Market factors calculated")

        return df

    def calculate_macro_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate macroeconomic factors (simplified version)
        In production, you would integrate with real macro data APIs

        Args:
            df: Price data DataFrame

        Returns:
            DataFrame with macro factors
        """
        logger.info("Adding macro factor placeholders")

        # Create placeholder macro factors based on date
        df['MACRO_NYSA_20_day_avg'] = df['returns'].rolling(20).mean()
        df['MACRO_USD_INDEX_signal'] = df['returns'].rolling(10).mean().apply(
            lambda x: 1 if x > 0 else -1
        )

        logger.info("Macro factors added")

        return df

    def _calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = series.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss.replace(0, 1)
        return 100 - (100 / (1 + rs))

    def get_feature_list(self) -> List[str]:
        """Get comprehensive list of features available in the data"""
        return [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'returns', 'log_returns',
            'MA_7', 'MA_14', 'MA_21', 'MA_30', 'MA_50', 'MA_200',
            'EMA_7', 'EMA_14', 'EMA_21', 'EMA_30', 'EMA_50', 'EMA_200',
            f'RSI_7', f'RSI_14', f'RSI_21', f'RSI_30', f'RSI_50',
            f'MA_ratio_7', f'MA_ratio_14', f'MA_ratio_21', f'MA_ratio_30', f'MA_ratio_50',
            'ROLLING_VOL_20', 'ROLLING_VOL_50',
            'BOLLINGER_BAND',
            f'ROC_1', f'ROC_2', f'ROC_3', f'ROC_5', f'ROC_10', f'ROC_20',
            'ATR',
            'VOLUME_MA_20', 'VOLUME_MA_50', 'VOLUME_DIFF', 'VOLUME_RATIO',
            'MACD', 'MACD_signal', 'MACD_diff',
            'RSI', 'KST', 'KST_signal',
            'CMF', 'OBV', 'Force_Index',
            'ROC_14',
            'Williams_R',
            'Stoch_K', 'Stoch_D',
            'DMI_plus', 'DMI_minus', 'ADX',
            'Aroon_Up', 'Aroon_Down',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
            'PPP_slope',
            'BENCHMARK_Close', 'BENCHMARK_returns', 'BENCHMARK_log_returns',
            'MARKET_beta', 'CROSS_RETURN',
            'MACRO_NYSA_20_day_avg', 'MACRO_USD_INDEX_signal'
        ]


if __name__ == "__main__":
    # Example usage
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
        'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS', 'ADBE', 'CRM', 'NFLX',
        'INTC', 'AMD', 'PYPL', 'BA', 'GOLD', 'SLV', 'XOM'
    ]

    start_date = datetime.strptime('2018-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2024-01-01', '%Y-%m-%d')

    collector = DataCollector(symbols, start_date, end_date)
    df = collector.fetch_data(interval='1d')

    # Add technical indicators
    df = collector.calculate_technical_indicators(df)

    # Add S&P 500 benchmark factors
    df = collector.calculate_market_factors(df, 'SPY')

    # Save results
    from config import RAW_DIR

    output_dir = RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / "stock_data.parquet")
    print(f"Data saved to {output_dir / 'stock_data.parquet'}")

    print(f"Final dataset columns: {len(df.columns)}")