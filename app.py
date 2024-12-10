import os
import math
import traceback
from functools import wraps
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from dotenv import load_dotenv
from flask import (Flask, jsonify, redirect, render_template, request,
                   session, url_for, flash, send_file)
from flask_bootstrap import Bootstrap
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from matplotlib import rc
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from werkzeug.security import check_password_hash
import zipfile
from io import BytesIO
from flask_socketio import SocketIO, emit
import eventlet

# **Matplotlibのバックエンドを'Agg'に設定（GUI不要）**
matplotlib.use('Agg')

# 環境変数をロード
load_dotenv()

app = Flask(__name__)

# 環境変数からSECRET_KEYを取得し、設定がなければエラーを発生
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set for Flask application")

# PostgreSQLの設定（環境変数から取得）
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///settings.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Flask-Migrateの設定
migrate = Migrate(app, db)

# Flask-Bootstrapの設定
bootstrap = Bootstrap(app)

# Flask-SocketIOの設定
socketio = SocketIO(app)

# 静的ファイルの保存ディレクトリ
APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # 現在のスクリプトファイルのディレクトリ
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'static', 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# パスワード認証用デコレーター
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 日本語フォント設定
rc('font', family='MS Gothic')


# データベースモデル
class UserSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    start_date = db.Column(db.String(10), nullable=False)
    end_date = db.Column(db.String(10), nullable=False)
    selected_indices = db.Column(db.String(200), nullable=False)
    spreads = db.Column(db.String(500), nullable=False)
    swaps_long = db.Column(db.String(500), nullable=False)
    swaps_short = db.Column(db.String(500), nullable=False)
    initial_margin = db.Column(db.Float, nullable=False)
    min_lot = db.Column(db.Float, nullable=False)
    max_lot = db.Column(db.Float, nullable=False)
    handle_small_hedge = db.Column(db.String(10), nullable=False)
    settlement_frequency = db.Column(db.String(10), nullable=False)
    optimize_weights = db.Column(db.Boolean, nullable=False)
    leverage = db.Column(db.String(500), nullable=False)  # 追加
    contract_size = db.Column(db.String(500), nullable=False)  # 追加

def get_data(start_date, end_date):
    # インデックスのティッカーシンボル
    indices = {
        'SP500': '^GSPC',
        'Nikkei': '^N225',
        'EuroStoxx': '^STOXX50E',
        'FTSE': '^FTSE',
        'DowJones': '^DJI',
        'AUS200': '^AXJO',
        'HK50': '^HSI',
        'SMI20': '^SSMI',
        'VIX': '^VIX'  # VIX指数を追加
    }
    
    # 為替レート
    fx_pairs = {
        'USDJPY': 'USDJPY=X',
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'AUDUSD': 'AUDUSD=X',
        'USDHKD': 'USDHKD=X',
        'USDCHF': 'USDCHF=X'
    }
    
    try:
        # データ取得
        data = {}
        for name, ticker in indices.items():
            data[name] = yf.download(ticker, start=start_date, end=end_date)
        
        # 為替レート取得
        fx_data = {}
        for name, ticker in fx_pairs.items():
            fx_data[name] = yf.download(ticker, start=start_date, end=end_date)

        # 通貨換算
        data['Nikkei']['Close_USD'] = data['Nikkei']['Close'] / fx_data['USDJPY']['Close']
        data['EuroStoxx']['Close_USD'] = data['EuroStoxx']['Close'] * fx_data['EURUSD']['Close']
        data['FTSE']['Close_USD'] = data['FTSE']['Close'] * fx_data['GBPUSD']['Close']
        data['AUS200']['Close_USD'] = data['AUS200']['Close'] * fx_data['AUDUSD']['Close']
        data['HK50']['Close_USD'] = data['HK50']['Close'] / fx_data['USDHKD']['Close']
        data['SMI20']['Close_USD'] = data['SMI20']['Close'] / fx_data['USDCHF']['Close']
        
        # リターン計算
        returns = pd.DataFrame()
        returns['SP500'] = np.log(data['SP500']['Close'] / data['SP500']['Close'].shift(1))
        returns['Nikkei'] = np.log(data['Nikkei']['Close_USD'] / data['Nikkei']['Close_USD'].shift(1))
        returns['EuroStoxx'] = np.log(data['EuroStoxx']['Close_USD'] / data['EuroStoxx']['Close_USD'].shift(1))
        returns['FTSE'] = np.log(data['FTSE']['Close_USD'] / data['FTSE']['Close_USD'].shift(1))
        returns['DowJones'] = np.log(data['DowJones']['Close'] / data['DowJones']['Close'].shift(1))
        returns['AUS200'] = np.log(data['AUS200']['Close_USD'] / data['AUS200']['Close_USD'].shift(1))
        returns['HK50'] = np.log(data['HK50']['Close_USD'] / data['HK50']['Close_USD'].shift(1))
        returns['SMI20'] = np.log(data['SMI20']['Close_USD'] / data['SMI20']['Close_USD'].shift(1))
        returns['VIX'] = np.log(data['VIX']['Close'] / data['VIX']['Close'].shift(1))  # VIXのリターンを追加
        
        # 価格データの整備
        price_data = pd.DataFrame()
        price_data['SP500'] = data['SP500']['Close']
        price_data['Nikkei'] = data['Nikkei']['Close_USD']
        price_data['EuroStoxx'] = data['EuroStoxx']['Close_USD']
        price_data['FTSE'] = data['FTSE']['Close_USD']
        price_data['DowJones'] = data['DowJones']['Close']
        price_data['AUS200'] = data['AUS200']['Close_USD']
        price_data['HK50'] = data['HK50']['Close_USD']
        price_data['SMI20'] = data['SMI20']['Close_USD']
        price_data['VIX'] = data['VIX']['Close']  # VIXの価格データを追加
        
        # データの確認用にインデックスを確認
        print("Price Data Index:", price_data.index)
        print("Returns Index:", returns.index)
        
        return price_data.dropna(), returns.dropna(), fx_data['USDJPY']['Close'].iloc[-1]
    except Exception as e:
        print(f"データ取得中にエラーが発生しました: {e}")
        traceback.print_exc()
        return None, None, None

def create_index_selection_frame():
    indices = {
        'Nikkei': 'Nikkei225',
        'EuroStoxx': 'EuroStoxx50',
        'FTSE': 'FTSE100',
        'DowJones': 'DowJones30',
        'AUS200': 'AUS200',
        'HK50': 'HK50',
        'SMI20': 'SMI20',
        'VIX': 'VIX'
    }
    return indices

def optimize_weights(returns, hedge_assets):
    """最小分散ポートフォリオのウェイトを計算"""
    def portfolio_variance(weights, returns_cov):
        return np.dot(weights.T, np.dot(returns_cov, weights))

    def constraint_sum(weights):
        return np.sum(weights) - 1.0

    n_assets = len(hedge_assets)
    initial_weights = np.array([1.0/n_assets] * n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraint = {'type': 'eq', 'fun': constraint_sum}
    
    cov_matrix = returns.cov()
    result = minimize(portfolio_variance, initial_weights,
                     args=(cov_matrix,),
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraint)
    
    return result.x if result.success else initial_weights

def calculate_hedge_ratios(returns, lookback_days, optimize_weights_flag, selected_assets, save_directory):
    hedge_assets = [asset for asset, is_selected in selected_assets.items() if is_selected]
    if not hedge_assets:
        raise ValueError("少なくとも1つの指数を選択してください。")

    hedge_ratios = pd.DataFrame(index=returns.index, columns=hedge_assets)

    for current_date in returns.index:
        lookback_start_date = current_date - pd.Timedelta(days=lookback_days)
        lookback_data = returns.loc[lookback_start_date:current_date]

        if len(lookback_data) < 2:
            hedge_ratios.loc[current_date] = [np.nan] * len(hedge_assets)
            continue

        valid_assets = hedge_assets.copy()
        while valid_assets:
            y = lookback_data['SP500']
            X = lookback_data[valid_assets]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            negative_assets = []
            for asset in valid_assets:
                if asset == 'VIX':
                    if model.params.get(asset, 0) > 0:
                        negative_assets.append(asset)
                else:
                    if model.params.get(asset, 0) < 0:
                        negative_assets.append(asset)

            if not negative_assets:
                if optimize_weights_flag:
                    weights = optimize_weights(lookback_data[valid_assets], valid_assets)
                else:
                    weights = np.array([1.0 / len(valid_assets)] * len(valid_assets))

                for i, asset in enumerate(valid_assets):
                    hedge_ratios.loc[current_date, asset] = model.params.get(asset, 0) * weights[i]

                excluded_assets = set(hedge_assets) - set(valid_assets)
                for asset in excluded_assets:
                    hedge_ratios.loc[current_date, asset] = 0.0

                break
            else:
                for asset in negative_assets:
                    valid_assets.remove(asset)

                if not valid_assets:
                    for asset in hedge_assets:
                        hedge_ratios.loc[current_date, asset] = 0.0

    # 欠損値を前日の値で埋める
    hedge_ratios = hedge_ratios.fillna(method='ffill')

    # 最初の行がNaNの場合はゼロで埋める
    hedge_ratios = hedge_ratios.fillna(0)

    hedge_ratios.to_csv(os.path.join(save_directory, "hedge_ratios.csv"), index=True)

    return hedge_ratios

def plot_correlation_matrix(data, save_directory):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title("Correlation Matrix of Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "correlation_matrix.png"))
    plt.close()

def plot_hedge_ratios(hedge_ratios, save_directory):
    plt.figure(figsize=(15, 8))
    for column in hedge_ratios.columns:
        plt.plot(hedge_ratios.index, hedge_ratios[column], label=column, marker='o', markersize=4)
    plt.title("Hedge Ratios Over Time")
    plt.xlabel("Date")
    plt.ylabel("Hedge Ratio")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "hedge_ratios.png"))
    plt.close()

def calculate_drawdowns(cum_returns):
    """全てのドローダウン指標を計算"""
    # 最大ドローダウン
    peak = np.maximum.accumulate(cum_returns)
    max_drawdowns = (cum_returns - peak) / peak
    
    # 相対ドローダウン（最大ドローダウンと同じ）
    relative_drawdowns = max_drawdowns.copy()
    
    # 絶対ドローダウン（初期値からの下落率）
    absolute_drawdowns = (cum_returns - cum_returns[0]) / cum_returns[0]
    
    return max_drawdowns, relative_drawdowns, absolute_drawdowns

def investment_simulation(price_data, hedge_ratios, costs, usdjpy_rate, settlement_frequency, save_directory):
    # ユーザーが設定したレバレッジと契約サイズを使用
    leverage = costs['leverage']
    contract_size = costs['contract_size']
    
    # コストをUSDに変換
    spreads_usd = {asset: value / usdjpy_rate for asset, value in costs['spreads'].items()}
    swaps_long_usd = {asset: value / usdjpy_rate for asset, value in costs['swaps_long'].items()}
    swaps_short_usd = {asset: value / usdjpy_rate for asset, value in costs['swaps_short'].items()}
    
    # 以下、既存のコードは同じ
    
    # コストをUSDに変換
    spreads_usd = {asset: value / usdjpy_rate for asset, value in costs['spreads'].items()}
    swaps_long_usd = {asset: value / usdjpy_rate for asset, value in costs['swaps_long'].items()}
    swaps_short_usd = {asset: value / usdjpy_rate for asset, value in costs['swaps_short'].items()}
    
    # リターン計算
    daily_returns = price_data.pct_change().dropna()

    # 決済期間の設定
    if settlement_frequency == 'quarterly':
        settlement_periods = pd.date_range(daily_returns.index[0], daily_returns.index[-1], freq='Q')
    elif settlement_frequency == 'monthly':
        settlement_periods = pd.date_range(daily_returns.index[0], daily_returns.index[-1], freq='M')
    else:
        settlement_periods = pd.date_range(daily_returns.index[0], daily_returns.index[-1], freq='Q')
        
    hedged_returns = []
    unhedged_returns = []
    
    for i in range(len(settlement_periods) - 1):
        start_date = settlement_periods[i]
        end_date = settlement_periods[i + 1]
        
        # 決済期間内のリターンデータを取得
        period_returns = daily_returns.loc[start_date:end_date]
        period_hedge_ratios = hedge_ratios.loc[start_date:end_date]
        
        if period_returns.empty:
            hedged_returns.append(0)
            unhedged_returns.append(0)
            continue
        
        # ヘッジされたリターンの計算
        # VIXはヘッジ比率がマイナスでロング、その他はヘッジ比率がプラスでショート
        adjusted_hedge_ratios = period_hedge_ratios.copy()
        for asset in adjusted_hedge_ratios.columns:
            if asset == 'VIX':
                adjusted_hedge_ratios[asset] = -adjusted_hedge_ratios[asset]
            else:
                adjusted_hedge_ratios[asset] = adjusted_hedge_ratios[asset]
        
        hedged_return = (period_returns['SP500'] + 
                         (period_returns[adjusted_hedge_ratios.columns] * adjusted_hedge_ratios.mean()).sum(axis=1) -
                         spreads_usd.get('SP500', 0) - swaps_short_usd.get('SP500', 0)).sum()
        hedged_returns.append(hedged_return)
        
        # アンヘッジされたリターンの計算
        unhedged_return = period_returns['SP500'].sum()
        unhedged_returns.append(unhedged_return)
    
    # 累積リターンの計算
    strategy_1_cum_return = np.cumprod(1 + np.array(hedged_returns)) - 1
    strategy_2_cum_return = np.cumprod(1 + np.array(unhedged_returns)) - 1
    
    # パフォーマンス指標の計算
    annualized_factor = 4 if settlement_frequency == 'quarterly' else 12  # 年率換算係数
    mean_return_1 = np.mean(hedged_returns) * annualized_factor
    mean_return_2 = np.mean(unhedged_returns) * annualized_factor
    
    vol_1 = np.std(hedged_returns) * np.sqrt(annualized_factor)
    vol_2 = np.std(unhedged_returns) * np.sqrt(annualized_factor)
    
    sharpe_ratio_1 = mean_return_1 / vol_1 if vol_1 != 0 else 0
    sharpe_ratio_2 = mean_return_2 / vol_2 if vol_2 != 0 else 0
    
    # ドローダウンの計算
    max_dd_1, rel_dd_1, abs_dd_1 = calculate_drawdowns(strategy_1_cum_return)
    max_dd_2, rel_dd_2, abs_dd_2 = calculate_drawdowns(strategy_2_cum_return)
    
    # 最大値の取得
    max_drawdown_1 = max_dd_1.min()
    max_drawdown_2 = max_dd_2.min()
    max_relative_dd_1 = rel_dd_1.min()
    max_relative_dd_2 = rel_dd_2.min()
    max_absolute_dd_1 = abs_dd_1.min()
    max_absolute_dd_2 = abs_dd_2.min()
    
    # 累積リターンのプロット
    plt.figure(figsize=(15, 8))
    plt.plot(settlement_periods[1:], strategy_1_cum_return, label="Multi-Asset Hedged")
    plt.plot(settlement_periods[1:], strategy_2_cum_return, label="Unhedged")
    plt.title("Cumulative Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "cumulative_returns.png"))
    plt.close()
    
    # 決済期間リターンのヒストグラム
    plt.figure(figsize=(15, 8))
    plt.hist(hedged_returns, bins=15, alpha=0.5, label="Multi-Asset Hedged")
    plt.hist(unhedged_returns, bins=15, alpha=0.5, label="Unhedged")
    plt.title(f"{settlement_frequency.capitalize()} Returns Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_directory, "returns_distribution.png"))
    plt.close()
    
    # ドローダウンのプロット
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 1, 1)
    plt.plot(settlement_periods[1:], max_dd_1, label="Multi-Asset Hedged", color='blue')
    plt.plot(settlement_periods[1:], max_dd_2, label="Unhedged", color='red')
    plt.title("Maximum Drawdown")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(settlement_periods[1:], rel_dd_1, label="Multi-Asset Hedged", color='blue')
    plt.plot(settlement_periods[1:], rel_dd_2, label="Unhedged", color='red')
    plt.title("Relative Drawdown")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(settlement_periods[1:], abs_dd_1, label="Absolute Drawdown", color='blue')
    plt.plot(settlement_periods[1:], abs_dd_2, label="Absolute Drawdown", color='red')
    plt.title("Absolute Drawdown")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "drawdowns.png"))
    plt.close()
    
    # パフォーマンス指標の計算と保存
    performance_metrics = pd.DataFrame({
        "Metric": ["Annual Return", "Annual Volatility", "Sharpe Ratio", 
                  "Maximum Drawdown", "Relative Drawdown", "Absolute Drawdown"],
        "Multi-Asset Hedged": [mean_return_1, vol_1, sharpe_ratio_1, 
                              max_drawdown_1, max_relative_dd_1, max_absolute_dd_1],
        "Unhedged": [mean_return_2, vol_2, sharpe_ratio_2, 
                     max_drawdown_2, max_relative_dd_2, max_absolute_dd_2]
    })
    
    # ドローダウンデータの保存
    drawdown_df = pd.DataFrame({
        f"{settlement_frequency.capitalize()}": [date.strftime('%Y-%m') for date in settlement_periods[1:]],
        "Multi-Asset Hedged (Maximum)": max_dd_1,
        "Unhedged (Maximum)": max_dd_2,
        "Multi-Asset Hedged (Relative)": rel_dd_1,
        "Unhedged (Relative)": rel_dd_2,
        "Multi-Asset Hedged (Absolute)": abs_dd_1,
        "Unhedged (Absolute)": abs_dd_2
    })
    
    # 結果の保存
    drawdown_df.to_csv(os.path.join(save_directory, f"{settlement_frequency}_drawdowns.csv"), index=False)
    results_df = pd.DataFrame({
        f"{settlement_frequency.capitalize()}": [date.strftime('%Y-%m') for date in settlement_periods[1:]],
        "Multi-Asset Hedged": hedged_returns,
        "Unhedged": unhedged_returns
    })
    results_df.to_csv(os.path.join(save_directory, f"{settlement_frequency}_returns.csv"), index=False)
    performance_metrics.to_csv(os.path.join(save_directory, f"{settlement_frequency}_performance_metrics.csv"), index=False)

def investment_simulation_2(price_data, hedge_ratios, costs, initial_margin, min_lot, max_lot, handle_small_hedge, settlement_frequency, save_directory, update_log):
    # ユーザーが設定したレバレッジと契約サイズを使用
    leverage = costs['leverage']
    lot_size_multiplier = costs['contract_size']  # contract_sizeをlot_size_multiplierとして使用
    
    # スプレッドとスワップコスト
    spreads = costs['spreads']
    swaps_long = costs['swaps_long'] 
    swaps_short = costs['swaps_short']
    
    # スプレッドとスワップコスト
    spreads = costs['spreads']
    swaps_long = costs['swaps_long'] 
    swaps_short = costs['swaps_short']

    # ヘッジ資産
    hedge_assets = hedge_ratios.columns

    # ロットサイズのリスト  
    lot_sizes = np.arange(min_lot, max_lot + 0.01, 0.01).round(2)
    lot_sizes = lot_sizes[lot_sizes >= 0.1]  # 最低ロットサイズを0.1に設定

    # 結果保存用リスト
    simulation_results = []

    # Position クラスの定義
    class Position:
        def __init__(self, initial_size, entry_price, entry_date):
            self.daily_size = initial_size
            self.cumulative_sizes = [initial_size]
            self.entry_prices = [entry_price]
            self.entry_dates = [entry_date]
            self.accumulated_swap_costs = 0
        
        def add_position(self, size, price, date):
            self.daily_size = size
            self.cumulative_sizes.append(size)
            self.entry_prices.append(price)
            self.entry_dates.append(date)
        
        @property
        def total_size(self):
            return sum(self.cumulative_sizes)

    # ロットサイズごとのシミュレーション
    for lot_size in lot_sizes:
        # 各ロットサイズ用のディレクトリを作成
        lot_dir = os.path.join(save_directory, f"lot_{lot_size}")
        os.makedirs(lot_dir, exist_ok=True)
        
        update_log(f"ロットサイズ {lot_size} の計算を開始します...")
        
        balance = initial_margin
        equity = initial_margin 
        usable_margin = initial_margin
        unrealized_swap_costs = 0

        # ポジション管理用の辞書
        positions = {'SP500': None}
        for asset in hedge_assets:
            positions[asset] = None

        hedge_lots = {asset: [] for asset in hedge_assets}
        balance_history = []
        equity_curve = []
        usable_margin_history = []
        
        # 決済頻度に基づく期間の設定
        if settlement_frequency == 'quarterly':
            settlement_periods = pd.date_range(price_data.index[0], price_data.index[-1], freq='Q')
        elif settlement_frequency == 'monthly':
            settlement_periods = pd.date_range(price_data.index[0], price_data.index[-1], freq='M')
        else:
            settlement_periods = pd.date_range(price_data.index[0], price_data.index[-1], freq='Q')

        current_settlement = 0
        last_settlement_date = None

        # 前のヘッジ比率を保持する変数
        previous_ratio = None

        # 各日のシミュレーション
        for date, row in price_data.iterrows():
            # 決済期間の終了日にポジションを決済
            if current_settlement < len(settlement_periods) and date >= settlement_periods[current_settlement]:
                realized_pnl = 0
                total_swap_costs = 0
                
                # 各ポジションの決済
                for asset, position in positions.items():
                    if position is not None:
                        current_price = row[asset]
                        position_pnl = 0
                        
                        # 価格変動によるPnL計算
                        for size, entry_price in zip(position.cumulative_sizes, position.entry_prices):
                            pnl = (current_price - entry_price) * size * lot_size_multiplier[asset]
                            if asset != 'SP500':
                                if asset != 'VIX':  # VIX以外のヘッジ資産は符号を反転
                                    pnl = -pnl
                            position_pnl += pnl

                        # スプレッドコスト計算
                        spread_cost = spreads.get(asset, 0.0) * abs(position.total_size)
                        
                        # スワップコスト計算（保有期間に基づく）
                        days_held = (date - position.entry_dates[0]).days
                        if asset == 'SP500' or asset == 'VIX':
                            swap_cost = swaps_long.get(asset, 0.0) * abs(position.total_size) * days_held
                        else:
                            swap_cost = swaps_short.get(asset, 0.0) * abs(position.total_size) * days_held
                        
                        total_swap_costs += swap_cost
                        realized_pnl += position_pnl - spread_cost - swap_cost
                        positions[asset] = None

                # 決済時のみbalanceを更新
                balance += realized_pnl
                equity = balance
                usable_margin = balance
                last_settlement_date = date
                current_settlement += 1

            # SP500のポジション設定と加算
            sp500_price = row['SP500']
            if positions['SP500'] is None:
                positions['SP500'] = Position(lot_size, sp500_price, date)
            else:
                positions['SP500'].add_position(lot_size, sp500_price, date)

            # ヘッジ比率の取得と更新
            # 日付が hedge_ratios に存在するか確認
            if date not in hedge_ratios.index:
                update_log(f"警告: hedge_ratios に日付 {date} が存在しません。前の値を使用します。")
                if previous_ratio is not None:
                    current_ratio = previous_ratio
                else:
                    current_ratio = hedge_ratios.iloc[0]
            else:
                current_ratio = hedge_ratios.loc[date]
                previous_ratio = current_ratio  # 前の値を保持

            # ヘッジポジションの更新と必要証拠金の計算
            hedge_margin = 0
            for asset in hedge_assets:
                ratio = current_ratio[asset]
                asset_price = row[asset]
                
                # 理想的なヘッジロットの計算
                if asset == 'VIX':
                    theoretical_hedge_lot = -lot_size * ratio * sp500_price / (asset_price * lot_size_multiplier[asset])
                else:
                    theoretical_hedge_lot = lot_size * ratio * sp500_price / asset_price
                
                # ヘッジロットの調整
                if abs(theoretical_hedge_lot) < min_lot:
                    if handle_small_hedge == 'short':
                        final_hedge_lot = math.ceil(abs(theoretical_hedge_lot) * 100) / 100
                        if asset == 'VIX':
                            final_hedge_lot = final_hedge_lot
                        else:
                            final_hedge_lot = -final_hedge_lot
                    else:
                        final_hedge_lot = 0.0
                else:
                    final_hedge_lot = math.ceil(abs(theoretical_hedge_lot) * 100) / 100
                    if theoretical_hedge_lot < 0:
                        final_hedge_lot = -final_hedge_lot

                # ポジション管理の更新（決済期間外）
                if positions[asset] is None and final_hedge_lot != 0:
                    positions[asset] = Position(final_hedge_lot, asset_price, date)
                elif positions[asset] is not None:
                    positions[asset].add_position(final_hedge_lot, asset_price, date)

                # 必要証拠金の計算
                if positions[asset] is not None:
                    position_value = abs(positions[asset].total_size * asset_price * lot_size_multiplier[asset])
                    hedge_margin += position_value / leverage.get(asset, 200)  # デフォルトレバレッジ200
            
                # ヘッジポジションの記録
                hedge_lots[asset].append(positions[asset].total_size if positions[asset] is not None else 0)

            # SP500の必要証拠金を計算
            sp500_margin = (positions['SP500'].total_size * sp500_price * lot_size_multiplier['SP500']) / leverage.get('SP500', 200)

            # 総必要証拠金
            required_margin = sp500_margin + hedge_margin

            # 含み損益の計算
            unrealized_pnl = 0
            for asset, position in positions.items():
                if position is not None:
                    current_price = row[asset]
                    for size, entry_price in zip(position.cumulative_sizes, position.entry_prices):
                        pnl = (current_price - entry_price) * size * lot_size_multiplier[asset]
                        if asset != 'SP500':
                            if asset != 'VIX':
                                pnl = -pnl
                        unrealized_pnl += pnl

            # 純資産（equity）と有効証拠金（usable_margin）の更新
            equity = balance + unrealized_pnl
            usable_margin = equity - required_margin

            # 記録の更新
            balance_history.append(balance)
            equity_curve.append(equity)
            usable_margin_history.append(usable_margin)

            # マージンチェック
            if usable_margin < 0:
                remaining_dates = len(price_data.index) - len(balance_history)
                balance_history.extend([0] * remaining_dates)
                equity_curve.extend([0] * remaining_dates)
                usable_margin_history.extend([0] * remaining_dates)
                
                for asset in hedge_assets:
                    hedge_lots[asset].extend([0.0] * remaining_dates)
                
                update_log(f"ロットサイズ {lot_size} でマージン不足が発生し、シミュレーションを中断しました。")
                break

        # シミュレーション結果の保存とプロット処理
        # ヘッジロットの記録をDataFrameに変換
        hedge_lots_df = pd.DataFrame(hedge_lots, index=price_data.index)
        hedge_lots_df.to_csv(os.path.join(lot_dir, "hedge_lots.csv"), index=True)

        # 財務指標の記録
        metrics_df = pd.DataFrame({
            "Date": price_data.index,
            "Balance": balance_history,
            "Equity": equity_curve,
            "Usable_Margin": usable_margin_history
        })
        metrics_df.to_csv(os.path.join(lot_dir, "financial_metrics.csv"), index=False)

        # ヘッジロットのプロット
        plt.figure(figsize=(15, 8))
        for asset in hedge_assets:
            plt.plot(hedge_lots_df.index, hedge_lots_df[asset], label=asset)
        plt.title(f"Hedge Lots Progression (Lot Size: {lot_size})")
        plt.xlabel("Date")
        plt.ylabel("Hedge Lots")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(lot_dir, "hedge_lots.png"))
        plt.close()

        # 財務指標のプロット
        plt.figure(figsize=(15, 8))
        plt.plot(metrics_df["Date"], metrics_df["Balance"], label="Settlement Balance", color='blue')
        plt.plot(metrics_df["Date"], metrics_df["Equity"], label="Daily Equity", color='green')
        plt.plot(metrics_df["Date"], metrics_df["Usable_Margin"], label="Daily Usable Margin", color='orange')
        plt.title(f"Financial Metrics (Lot Size: {lot_size})")
        plt.xlabel("Date")
        plt.ylabel("USD")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(lot_dir, "financial_metrics.png"))
        plt.close()

        # パフォーマンス指標の計算と結果の保存
        final_metrics = {
            "Lot Size": lot_size,
            "Final Balance": balance,
            "Final Equity": equity,
            "Final Usable Margin": usable_margin,
            "Max Drawdown": min(equity_curve) / max(equity_curve) - 1 if max(equity_curve) != 0 else -1,
            "Sharpe Ratio": (np.mean(np.diff(equity_curve)) / np.std(np.diff(equity_curve))) * np.sqrt(252) if np.std(np.diff(equity_curve)) != 0 else 0
        }
        simulation_results.append(final_metrics)

    # 全シミュレーション結果の保存
    results_df = pd.DataFrame(simulation_results)
    results_df.to_csv(os.path.join(save_directory, "simulation_results.csv"), index=False)

    # 最適なロットサイズの結果を表示
    valid_results = results_df[results_df["Final Usable Margin"] >= 0]
    if not valid_results.empty:
        best_result = valid_results.sort_values("Final Balance", ascending=False).iloc[0]
        update_log(f"\n最適なロットサイズ: {best_result['Lot Size']}")
        update_log(f"最終残高: {best_result['Final Balance']:.2f} USD")
        update_log(f"最終純資産: {best_result['Final Equity']:.2f} USD")
        update_log(f"最終有効証拠金: {best_result['Final Usable Margin']:.2f} USD")
    else:
        update_log("\n有効な結果が見つかりませんでした。")

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        # 環境変数からハッシュ化されたパスワードを取得
        stored_password_hash = os.getenv('PASSWORD_HASH')
        if not stored_password_hash:
            flash('サーバー設定に問題があります。')
            return render_template('login.html')
        if check_password_hash(stored_password_hash, password):
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            flash('パスワードが正しくありません。')
            return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        submit_action = request.form.get('submit_action')

        print(f"Received submit_action: {submit_action}")
        print(f"Form Data: {request.form}")

        if submit_action == 'save':
            try:
                start_date = request.form.get('start_date', '2000-01-01')
                end_date = request.form.get('end_date', (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'))
                period = request.form.get('period', '3か月')
                optimize_weights_flag = True if request.form.get('optimize_weights') == 'on' else False

                selected_indices = request.form.getlist('indices')
                costs_spread = request.form.getlist('spread')
                costs_swaps_long = request.form.getlist('swap_long')
                costs_swaps_short = request.form.getlist('swap_short')
                setting_name = request.form.get('setting_name', 'default')

                initial_margin = float(request.form.get('initial_margin', '2000'))
                min_lot = float(request.form.get('min_lot', '0.10'))
                max_lot = float(request.form.get('max_lot', '0.15'))
                handle_small_hedge = request.form.get('handle_small_hedge', 'short')
                settlement_frequency = request.form.get('settlement_frequency', 'monthly')

                asset_names = ['SP500', 'Nikkei', 'EuroStoxx', 'FTSE', 'DowJones', 'AUS200', 'HK50', 'SMI20', 'VIX']
                costs = {
                    'spreads': {},
                    'swaps_long': {},
                    'swaps_short': {},
                    'leverage': {},
                    'contract_size': {}
                }

                for i, asset in enumerate(asset_names):
                    try:
                        costs['spreads'][asset] = float(costs_spread[i])
                        costs['swaps_long'][asset] = float(costs_swaps_long[i])
                        costs['swaps_short'][asset] = float(costs_swaps_short[i])
                    except (IndexError, ValueError):
                        costs['spreads'][asset] = 0.0
                        costs['swaps_long'][asset] = 0.0
                        costs['swaps_short'][asset] = 0.0

                    try:
                        costs['leverage'][asset] = float(request.form.get(f'leverage_{asset}'))
                        costs['contract_size'][asset] = float(request.form.get(f'contract_size_{asset}'))
                    except (TypeError, ValueError):
                        flash(f'{asset}のレバレッジまたは契約サイズが無効です。デフォルト値を使用します。')
                        costs['leverage'][asset] = 200.0 if asset != 'VIX' else 100.0
                        costs['contract_size'][asset] = 1.0 if asset != 'VIX' else 100.0

                existing_setting = UserSetting.query.filter_by(name=setting_name).first()
                if existing_setting:
                    existing_setting.start_date = start_date
                    existing_setting.end_date = end_date
                    existing_setting.selected_indices = ','.join(selected_indices)
                    existing_setting.spreads = ','.join([str(costs['spreads'][asset]) for asset in asset_names])
                    existing_setting.swaps_long = ','.join([str(costs['swaps_long'][asset]) for asset in asset_names])
                    existing_setting.swaps_short = ','.join([str(costs['swaps_short'][asset]) for asset in asset_names])
                    existing_setting.initial_margin = initial_margin
                    existing_setting.min_lot = min_lot
                    existing_setting.max_lot = max_lot
                    existing_setting.handle_small_hedge = handle_small_hedge
                    existing_setting.settlement_frequency = settlement_frequency
                    existing_setting.optimize_weights = optimize_weights_flag
                    existing_setting.leverage = ','.join([str(costs['leverage'][asset]) for asset in asset_names])
                    existing_setting.contract_size = ','.join([str(costs['contract_size'][asset]) for asset in asset_names])
                else:
                    new_setting = UserSetting(
                        name=setting_name,
                        start_date=start_date,
                        end_date=end_date,
                        selected_indices=','.join(selected_indices),
                        spreads=','.join([str(costs['spreads'][asset]) for asset in asset_names]),
                        swaps_long=','.join([str(costs['swaps_long'][asset]) for asset in asset_names]),
                        swaps_short=','.join([str(costs['swaps_short'][asset]) for asset in asset_names]),
                        initial_margin=initial_margin,
                        min_lot=min_lot,
                        max_lot=max_lot,
                        handle_small_hedge=handle_small_hedge,
                        settlement_frequency=settlement_frequency,
                        optimize_weights=optimize_weights_flag,
                        leverage=','.join([str(costs['leverage'][asset]) for asset in asset_names]),
                        contract_size=','.join([str(costs['contract_size'][asset]) for asset in asset_names])
                    )
                    db.session.add(new_setting)
                db.session.commit()
                return jsonify({'success': True})

            except Exception as e:
                print(f'エラーが発生しました: {str(e)}')
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

        elif submit_action == 'analyze':
            try:
                start_date = request.form.get('start_date', '2000-01-01')
                end_date = request.form.get('end_date', (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'))
                period = request.form.get('period', '3か月')
                optimize_weights_flag = True if request.form.get('optimize_weights') == 'on' else False

                selected_indices = request.form.getlist('indices')
                costs_spread = request.form.getlist('spread')
                costs_swaps_long = request.form.getlist('swap_long')
                costs_swaps_short = request.form.getlist('swap_short')

                initial_margin = float(request.form.get('initial_margin', '2000'))
                min_lot = float(request.form.get('min_lot', '0.10'))
                max_lot = float(request.form.get('max_lot', '0.15'))
                handle_small_hedge = request.form.get('handle_small_hedge', 'short')
                settlement_frequency = request.form.get('settlement_frequency', 'monthly')

                asset_names = ['SP500', 'Nikkei', 'EuroStoxx', 'FTSE', 'DowJones', 'AUS200', 'HK50', 'SMI20', 'VIX']
                costs = {
                    'spreads': {},
                    'swaps_long': {},
                    'swaps_short': {},
                    'leverage': {},
                    'contract_size': {}
                }

                for i, asset in enumerate(asset_names):
                    try:
                        costs['spreads'][asset] = float(costs_spread[i])
                        costs['swaps_long'][asset] = float(costs_swaps_long[i])
                        costs['swaps_short'][asset] = float(costs_swaps_short[i])
                    except (IndexError, ValueError):
                        costs['spreads'][asset] = 0.0
                        costs['swaps_long'][asset] = 0.0
                        costs['swaps_short'][asset] = 0.0

                    try:
                        costs['leverage'][asset] = float(request.form.get(f'leverage_{asset}'))
                        costs['contract_size'][asset] = float(request.form.get(f'contract_size_{asset}'))
                    except (TypeError, ValueError):
                        flash(f'{asset}のレバレッジまたは契約サイズが無効です。デフォルト値を使用します。')
                        costs['leverage'][asset] = 200.0 if asset != 'VIX' else 100.0
                        costs['contract_size'][asset] = 1.0 if asset != 'VIX' else 100.0

                socketio.emit('log', {'message': 'データの取得を開始します...'})
                price_data, returns, usdjpy_rate = get_data(start_date, end_date)

                if price_data is None or returns is None or usdjpy_rate is None:
                    socketio.emit('log', {'message': 'データ取得に失敗しました。'})
                    return jsonify({'error': 'データ取得に失敗しました'}), 500

                socketio.emit('log', {'message': '相関行列のプロットを作成します...'})
                plot_correlation_matrix(returns, OUTPUT_FOLDER)

                period_mapping = {
                    '30日': 30,
                    '3か月': 90,
                    '半年': 180,
                    '1年': 365
                }
                lookback_days = period_mapping.get(period, 90)

                socketio.emit('log', {'message': 'ヘッジ比率の計算を開始します...'})
                selected_assets = {asset: (asset in selected_indices) for asset in ['Nikkei', 'EuroStoxx', 'FTSE', 'DowJones', 'AUS200', 'HK50', 'SMI20', 'VIX']}
                hedge_ratios = calculate_hedge_ratios(returns, lookback_days, optimize_weights_flag, selected_assets, OUTPUT_FOLDER)

                socketio.emit('log', {'message': 'ヘッジ比率のプロットを作成します...'})
                plot_hedge_ratios(hedge_ratios, OUTPUT_FOLDER)

                socketio.emit('log', {'message': '投資シミュレーション1を開始します...'})
                investment_simulation(price_data, hedge_ratios, costs, usdjpy_rate, settlement_frequency, OUTPUT_FOLDER)

                def log_callback(message):
                    socketio.emit('log', {'message': message})

                socketio.emit('log', {'message': '投資シミュレーション2を開始します...'})
                investment_simulation_2(price_data, hedge_ratios, costs, initial_margin, min_lot, max_lot, 
                                     handle_small_hedge, settlement_frequency, OUTPUT_FOLDER, log_callback)

                socketio.emit('log', {'message': '分析とシミュレーションが完了しました。'})
                return jsonify({'redirect': url_for('results')})

            except Exception as e:
                print(f'エラーが発生しました: {str(e)}')
                traceback.print_exc()
                socketio.emit('log', {'message': f'エラーが発生しました: {str(e)}'})
                return jsonify({'error': str(e)}), 500

        else:
            return jsonify({'error': '不正なアクションが指定されました'}), 400

    settings = UserSetting.query.all()
    indices = create_index_selection_frame()
    leverage = {}
    contract_size = {}
    for setting in settings:
        leverage_values = setting.leverage.split(',')
        contract_size_values = setting.contract_size.split(',')
        asset_names = ['SP500', 'Nikkei', 'EuroStoxx', 'FTSE', 'DowJones', 'AUS200', 'HK50', 'SMI20', 'VIX']
        for i, asset in enumerate(asset_names):
            leverage[asset] = float(leverage_values[i]) if i < len(leverage_values) else 200.0
            contract_size[asset] = float(contract_size_values[i]) if i < len(contract_size_values) else 1.0
    return render_template('index.html', indices=indices, settings=settings, 
                         end_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                         leverage=leverage, contract_size=contract_size)

@app.route('/load_setting/<int:setting_id>', methods=['GET'])
@login_required
def load_setting(setting_id):
    setting = UserSetting.query.get_or_404(setting_id)
    asset_names = ['SP500', 'Nikkei', 'EuroStoxx', 'FTSE', 'DowJones', 'AUS200', 'HK50', 'SMI20', 'VIX']
    leverage_values = setting.leverage.split(',')
    contract_size_values = setting.contract_size.split(',')
    leverage = {asset: leverage_values[i] for i, asset in enumerate(asset_names) if i < len(leverage_values)}
    contract_size = {asset: contract_size_values[i] for i, asset in enumerate(asset_names) if i < len(contract_size_values)}
    data = {
        'name': setting.name,
        'start_date': setting.start_date,
        'end_date': setting.end_date,
        'selected_indices': setting.selected_indices.split(','),
        'spreads': setting.spreads.split(','),
        'swaps_long': setting.swaps_long.split(','),
        'swaps_short': setting.swaps_short.split(','),
        'initial_margin': setting.initial_margin,
        'min_lot': setting.min_lot,
        'max_lot': setting.max_lot,
        'handle_small_hedge': setting.handle_small_hedge,
        'settlement_frequency': setting.settlement_frequency,
        'optimize_weights': setting.optimize_weights,
        'leverage': leverage,
        'contract_size': contract_size
    }
    return jsonify(data)

@app.route('/delete_setting/<int:setting_id>', methods=['POST'])
@login_required
def delete_setting(setting_id):
    setting = UserSetting.query.get_or_404(setting_id)
    db.session.delete(setting)
    db.session.commit()
    flash('設定を削除しました。')
    return jsonify({'status': 'success'})

@app.route('/results')
@login_required
def results():
    save_directory = OUTPUT_FOLDER
    if not os.path.exists(save_directory):
        flash('結果ディレクトリが存在しません。')
        return redirect(url_for('index'))

    # 保存されたファイルのリスト（サブディレクトリ含む）
    images = {}
    csvs = {}
    lot_sizes = []

    for root, dirs, files in os.walk(save_directory):
        for file in files:
            if file.endswith('.png'):
                # 相関行列や全体のプロット
                if file not in ['correlation_matrix.png', 'hedge_ratios.png', 'cumulative_returns.png', 'returns_distribution.png', 'drawdowns.png']:
                    # ロットサイズごとの画像
                    # 例: lot_0.10/hedge_lots.png を key: 'lot_0.10_hedge_lots.png'
                    subdir = os.path.basename(root)
                    if subdir.startswith('lot_'):
                        lot_size = subdir.split('_')[1]
                        lot_sizes.append(float(lot_size))
                        if file == 'hedge_lots.png':
                            key = f"lot_{lot_size}_hedge_lots.png"
                        elif file == 'financial_metrics.png':
                            key = f"lot_{lot_size}_financial_metrics.png"
                        else:
                            key = f"lot_{lot_size}_{file}"
                        images[key] = url_for('static', filename=f'output/{subdir}/{file}')
                else:
                    images[file] = url_for('static', filename=f'output/{file}')
            elif file.endswith('.csv'):
                # 相関行列や全体のCSV
                if file not in ['hedge_ratios.csv', 'cumulative_returns.csv', 'returns_distribution.csv', 'drawdowns.csv', 'performance_metrics.csv', 'simulation_results.csv']:
                    # ロットサイズごとのCSV
                    subdir = os.path.basename(root)
                    if subdir.startswith('lot_'):
                        lot_size = subdir.split('_')[1]
                        lot_sizes.append(float(lot_size))
                        if file == 'hedge_lots.csv':
                            key = f"lot_{lot_size}_hedge_lots.csv"
                        elif file == 'financial_metrics.csv':
                            key = f"lot_{lot_size}_financial_metrics.csv"
                        else:
                            key = f"lot_{lot_size}_{file}"
                        csvs[key] = url_for('static', filename=f'output/{subdir}/{file}')
                else:
                    csvs[file] = url_for('static', filename=f'output/{file}')

    # 重複するロットサイズを削除し、ソート
    lot_sizes = sorted(list(set(lot_sizes)))

    # ヘッジ資産のリストを作成
    hedge_assets = ['SP500', 'Nikkei', 'EuroStoxx', 'FTSE', 'DowJones', 'AUS200', 'HK50', 'SMI20', 'VIX']

    # JavaScriptに渡すデータを準備
    js_data = {
        'lot_sizes': lot_sizes,
        'hedge_assets': hedge_assets,
        'images': images,
        'csvs': csvs,
    }

    # テンプレートにデータを渡して表示
    return render_template('results.html', 
                         images=images, 
                         csvs=csvs, 
                         lot_sizes=lot_sizes,
                         hedge_assets=hedge_assets,
                         js_data=js_data)

@app.route('/download_lot_results/<float:lot_size>', methods=['POST'])
@login_required
def download_lot_results(lot_size):
    subdir = f"lot_{lot_size}"
    lot_dir = os.path.join(OUTPUT_FOLDER, subdir)
    if not os.path.exists(lot_dir):
        flash(f"ロットサイズ {lot_size} のディレクトリが存在しません。")
        return redirect(url_for('results'))

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(lot_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # アーカイブ内のパスをサブディレクトリ名とファイル名に限定
                arcname = f"{file}"
                zf.write(file_path, arcname)

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'lot_size_{lot_size}_results.zip'
    )

@socketio.on('start_analysis', namespace='/analysis')
def handle_analysis(data):
   try:
       def log_callback(message):
           emit('log', {'message': message}, namespace='/analysis')

       investment_simulation_2(data['price_data'], data['hedge_ratios'], 
                             data['costs'], data['initial_margin'], 
                             data['min_lot'], data['max_lot'], 
                             data['handle_small_hedge'], 
                             data['settlement_frequency'], 
                             data['save_directory'], 
                             log_callback)
                             
       emit('analysis_complete', {'status': 'success'}, namespace='/analysis')
   except Exception as e:
       emit('analysis_error', {'error': str(e)}, namespace='/analysis')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)
