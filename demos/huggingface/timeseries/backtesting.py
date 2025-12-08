#!/usr/bin/env python3
"""
Backtesting Demo

This demo shows how to:
- Implement a simple trading strategy
- Replay historical data from Syna
- Calculate performance metrics
- Show latency metrics for real-time suitability

Run with: python backtesting.py

Requirements: numpy
"""

import os
import sys
import time
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from Syna import SynaDB


class PriceSimulator:
    """Simulates realistic price data with trends and volatility."""
    
    def __init__(self, initial_price: float = 100.0, volatility: float = 0.02):
        self.price = initial_price
        self.volatility = volatility
        self.trend = 0.0
        self.trend_duration = 0
    
    def next_price(self) -> float:
        """Generate next price using geometric Brownian motion with trends."""
        # Occasionally change trend
        if self.trend_duration <= 0:
            self.trend = random.gauss(0, 0.001)
            self.trend_duration = random.randint(50, 200)
        self.trend_duration -= 1
        
        # Random walk with drift
        return_rate = self.trend + random.gauss(0, self.volatility)
        self.price *= (1 + return_rate)
        
        return self.price


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: int
    action: str  # 'buy' or 'sell'
    price: float
    quantity: float
    
    @property
    def value(self) -> float:
        return self.price * self.quantity


@dataclass
class Position:
    """Represents current position."""
    quantity: float = 0.0
    avg_price: float = 0.0
    
    def update(self, trade: Trade):
        if trade.action == 'buy':
            total_cost = self.quantity * self.avg_price + trade.value
            self.quantity += trade.quantity
            self.avg_price = total_cost / self.quantity if self.quantity > 0 else 0
        else:  # sell
            self.quantity -= trade.quantity
            if self.quantity <= 0:
                self.quantity = 0
                self.avg_price = 0


class TradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def signal(self, prices: np.ndarray, current_position: float) -> Optional[str]:
        """Return 'buy', 'sell', or None."""
        raise NotImplementedError


class MovingAverageCrossover(TradingStrategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__(f"MA_Crossover({short_window},{long_window})")
        self.short_window = short_window
        self.long_window = long_window
        self.prev_short_ma = None
        self.prev_long_ma = None
    
    def signal(self, prices: np.ndarray, current_position: float) -> Optional[str]:
        if len(prices) < self.long_window:
            return None
        
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        signal = None
        
        if self.prev_short_ma is not None and self.prev_long_ma is not None:
            # Golden cross: short MA crosses above long MA
            if self.prev_short_ma <= self.prev_long_ma and short_ma > long_ma:
                if current_position <= 0:
                    signal = 'buy'
            # Death cross: short MA crosses below long MA
            elif self.prev_short_ma >= self.prev_long_ma and short_ma < long_ma:
                if current_position > 0:
                    signal = 'sell'
        
        self.prev_short_ma = short_ma
        self.prev_long_ma = long_ma
        
        return signal


class MeanReversion(TradingStrategy):
    """Mean reversion strategy using Bollinger Bands."""
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(f"MeanReversion({window},{num_std})")
        self.window = window
        self.num_std = num_std
    
    def signal(self, prices: np.ndarray, current_position: float) -> Optional[str]:
        if len(prices) < self.window:
            return None
        
        window_prices = prices[-self.window:]
        mean = np.mean(window_prices)
        std = np.std(window_prices)
        
        current_price = prices[-1]
        upper_band = mean + self.num_std * std
        lower_band = mean - self.num_std * std
        
        # Buy when price drops below lower band (oversold)
        if current_price < lower_band and current_position <= 0:
            return 'buy'
        # Sell when price rises above upper band (overbought)
        elif current_price > upper_band and current_position > 0:
            return 'sell'
        
        return None


class Backtester:
    """Backtesting engine that replays historical data."""
    
    def __init__(self, strategy: TradingStrategy, initial_capital: float = 10000.0):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = Position()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.prices_seen: List[float] = []
    
    def run(self, prices: np.ndarray) -> dict:
        """Run backtest on historical prices."""
        self.capital = self.initial_capital
        self.position = Position()
        self.trades = []
        self.equity_curve = []
        self.prices_seen = []
        
        for i, price in enumerate(prices):
            self.prices_seen.append(price)
            
            # Get signal from strategy
            signal = self.strategy.signal(
                np.array(self.prices_seen), 
                self.position.quantity
            )
            
            # Execute trade if signal
            if signal == 'buy' and self.capital > 0:
                # Buy with all available capital
                quantity = self.capital / price
                trade = Trade(
                    timestamp=i,
                    action='buy',
                    price=price,
                    quantity=quantity
                )
                self.trades.append(trade)
                self.position.update(trade)
                self.capital = 0
                
            elif signal == 'sell' and self.position.quantity > 0:
                # Sell entire position
                trade = Trade(
                    timestamp=i,
                    action='sell',
                    price=price,
                    quantity=self.position.quantity
                )
                self.trades.append(trade)
                self.capital = trade.value
                self.position.update(trade)
            
            # Calculate current equity
            equity = self.capital + self.position.quantity * price
            self.equity_curve.append(equity)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Total return
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return (assuming daily data)
        n_periods = len(equity)
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        if len(self.trades) >= 2:
            profits = []
            for i in range(0, len(self.trades) - 1, 2):
                if i + 1 < len(self.trades):
                    buy = self.trades[i]
                    sell = self.trades[i + 1]
                    if buy.action == 'buy' and sell.action == 'sell':
                        profit = (sell.price - buy.price) / buy.price
                        profits.append(profit)
            
            win_rate = sum(1 for p in profits if p > 0) / len(profits) if profits else 0
            avg_profit = np.mean(profits) if profits else 0
        else:
            win_rate = 0
            avg_profit = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit,
            'final_equity': equity[-1]
        }


def main():
    print("=== Backtesting Demo ===\n")
    
    import tempfile
    import shutil
    
    # Use temporary directory for database
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "backtesting.db")
    
    try:
        # 1. Generate historical price data
        print("1. Generating historical price data...")
        simulator = PriceSimulator(initial_price=100.0, volatility=0.02)
        num_days = 1000  # ~4 years of daily data
        
        with SynaDB(db_path) as db:
            for i in range(num_days):
                price = simulator.next_price()
                db.put_float("prices/asset1", price)
        
        print(f"   ✓ Generated {num_days} price points\n")
        
        # 2. Load historical data from Syna
        print("2. Loading historical data from Syna...")
        
        start_time = time.time()
        with SynaDB(db_path) as db:
            prices = db.get_history_tensor("prices/asset1")
        load_time = time.time() - start_time
        
        print(f"   ✓ Loaded {len(prices)} prices in {load_time * 1000:.2f}ms")
        print(f"   Price range: [{prices.min():.2f}, {prices.max():.2f}]")
        print(f"   Start: {prices[0]:.2f}, End: {prices[-1]:.2f}")
        print(f"   Buy & Hold return: {(prices[-1] - prices[0]) / prices[0]:.2%}\n")
        
        # 3. Backtest Moving Average Crossover strategy
        print("3. Backtesting Moving Average Crossover strategy...")
        
        ma_strategy = MovingAverageCrossover(short_window=10, long_window=30)
        backtester = Backtester(ma_strategy, initial_capital=10000.0)
        
        start_time = time.time()
        ma_metrics = backtester.run(prices)
        backtest_time = time.time() - start_time
        
        print(f"   Strategy: {ma_strategy.name}")
        print(f"   Backtest time: {backtest_time * 1000:.2f}ms")
        print(f"   Results:")
        print(f"      Total Return: {ma_metrics['total_return']:.2%}")
        print(f"      Annualized Return: {ma_metrics['annualized_return']:.2%}")
        print(f"      Volatility: {ma_metrics['volatility']:.2%}")
        print(f"      Sharpe Ratio: {ma_metrics['sharpe_ratio']:.2f}")
        print(f"      Max Drawdown: {ma_metrics['max_drawdown']:.2%}")
        print(f"      Number of Trades: {ma_metrics['num_trades']}")
        print(f"      Win Rate: {ma_metrics['win_rate']:.2%}")
        print(f"      Final Equity: ${ma_metrics['final_equity']:.2f}\n")
        
        # 4. Backtest Mean Reversion strategy
        print("4. Backtesting Mean Reversion strategy...")
        
        mr_strategy = MeanReversion(window=20, num_std=2.0)
        backtester = Backtester(mr_strategy, initial_capital=10000.0)
        
        start_time = time.time()
        mr_metrics = backtester.run(prices)
        backtest_time = time.time() - start_time
        
        print(f"   Strategy: {mr_strategy.name}")
        print(f"   Backtest time: {backtest_time * 1000:.2f}ms")
        print(f"   Results:")
        print(f"      Total Return: {mr_metrics['total_return']:.2%}")
        print(f"      Annualized Return: {mr_metrics['annualized_return']:.2%}")
        print(f"      Volatility: {mr_metrics['volatility']:.2%}")
        print(f"      Sharpe Ratio: {mr_metrics['sharpe_ratio']:.2f}")
        print(f"      Max Drawdown: {mr_metrics['max_drawdown']:.2%}")
        print(f"      Number of Trades: {mr_metrics['num_trades']}")
        print(f"      Win Rate: {mr_metrics['win_rate']:.2%}")
        print(f"      Final Equity: ${mr_metrics['final_equity']:.2f}\n")
        
        # 5. Compare strategies
        print("5. Strategy comparison:")
        print(f"   {'Metric':<25} {'MA Crossover':>15} {'Mean Reversion':>15}")
        print(f"   {'-' * 55}")
        print(f"   {'Total Return':<25} {ma_metrics['total_return']:>14.2%} {mr_metrics['total_return']:>14.2%}")
        print(f"   {'Sharpe Ratio':<25} {ma_metrics['sharpe_ratio']:>15.2f} {mr_metrics['sharpe_ratio']:>15.2f}")
        print(f"   {'Max Drawdown':<25} {ma_metrics['max_drawdown']:>14.2%} {mr_metrics['max_drawdown']:>14.2%}")
        print(f"   {'Number of Trades':<25} {ma_metrics['num_trades']:>15} {mr_metrics['num_trades']:>15}")
        print()
        
        # 6. Benchmark replay latency
        print("6. Benchmarking replay latency...")
        
        latencies = []
        ma_strategy = MovingAverageCrossover(short_window=10, long_window=30)
        
        for _ in range(100):
            start = time.time()
            # Simulate single-step replay
            signal = ma_strategy.signal(prices[:100], 0)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        latencies = np.array(latencies)
        print(f"   Single-step signal generation (100 iterations):")
        print(f"      Mean: {np.mean(latencies):.4f}ms")
        print(f"      P50:  {np.percentile(latencies, 50):.4f}ms")
        print(f"      P95:  {np.percentile(latencies, 95):.4f}ms")
        print(f"      P99:  {np.percentile(latencies, 99):.4f}ms")
        print(f"      Suitable for real-time: {'Yes' if np.percentile(latencies, 99) < 1.0 else 'No'}\n")
        
        # 7. Store backtest results
        print("7. Storing backtest results to Syna...")
        
        with SynaDB(db_path) as db:
            # Store equity curves
            for i, equity in enumerate(backtester.equity_curve):
                db.put_float("backtest/equity", equity)
            
            # Store metrics
            for key, value in ma_metrics.items():
                if isinstance(value, (int, float)):
                    db.put_float(f"backtest/metrics/ma/{key}", float(value))
            
            for key, value in mr_metrics.items():
                if isinstance(value, (int, float)):
                    db.put_float(f"backtest/metrics/mr/{key}", float(value))
        
        print(f"   ✓ Stored equity curve ({len(backtester.equity_curve)} points)")
        print(f"   ✓ Stored performance metrics\n")
        
        # 8. Show storage statistics
        print("8. Storage statistics:")
        file_size = os.path.getsize(db_path)
        print(f"   Database size: {file_size / 1024:.2f} KB")
        print(f"   Bytes per price point: {file_size / num_days:.1f}")
        
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
    
    print("=== Demo Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

