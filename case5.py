import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import warnings
import time
import re

warnings.filterwarnings('ignore')


# 纯Pandas实现的指标计算函数
def pandas_MACD(close, fast=12, slow=26, signal=9):
    """使用Pandas实现MACD指标计算"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def pandas_RSI(close, period=14):
    """使用Pandas实现RSI指标计算"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # 处理除零情况
    rs = avg_gain / avg_loss.replace(0, np.nan).ffill().bfill()
    rsi = 100 - (100 / (1 + rs))
    return rsi


def pandas_KDJ(high, low, close, window=14):
    """使用Pandas实现KDJ指标计算，带除零保护"""
    min_low = low.rolling(window=window, min_periods=1).min()
    max_high = high.rolling(window=window, min_periods=1).max()

    # 除零保护
    diff = max_high - min_low
    diff = diff.replace(0, 1)  # 避免除零错误

    rsv = (close - min_low) / diff * 100
    k = rsv.ewm(alpha=1 / 3).mean()
    d = k.ewm(alpha=1 / 3).mean()
    j = 3 * k - 2 * d
    return k, d, j


def pandas_BOLL(close, window=20, num_std=2):
    """使用Pandas实现布林带指标计算"""
    mid = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


class JiuPinShenLong:
    """
    九品神龙股票分析工具 - 增强版
    功能包含：
    1. 多周期均线系统（5/10/20/60/120/250日）
    2. 成交量异动分析（量比、量能潮）
    3. MACD、KDJ、RSI多指标共振（纯Pandas实现）
    4. 趋势通道与布林带分析
    5. 主力资金流向分析
    6. 买卖信号生成与仓位管理
    7. 可视化分析图表
    8. 策略回测与绩效评估
    9. 自动获取龙头股并筛选符合条件的股票
    """

    def __init__(self, stock_code, start_date=None, end_date=None):
        """
        初始化分析工具
        :param stock_code: 股票代码 (e.g. '600000' 或 '000001')
        :param start_date: 开始日期 (YYYY-MM-DD)
        :param end_date: 结束日期 (YYYY-MM-DD)
        """
        self.stock_code = stock_code
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.today() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
        self.data = self._get_stock_data()
        if not self.data.empty:
            self._calculate_indicators()
        else:
            print(f"警告: 未获取到股票 {stock_code} 的数据")

    def _get_stock_data(self):
        """使用AKShare获取股票数据（兼容最新接口）"""
        # 尝试使用stock_zh_a_daily接口
        try:
            # 2025年最新AKShare接口
            symbol = f"sh{self.stock_code}" if self.stock_code.startswith('6') else f"sz{self.stock_code}"
            df = ak.stock_zh_a_daily(symbol=symbol,
                                     start_date=self.start_date,
                                     end_date=self.end_date,
                                     adjust="hfq")
            # 重命名列以统一格式
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'amount': 'amount'
            })
        except Exception as e:
            print(f"使用stock_zh_a_daily接口失败: {e}")
            # 回退到备用接口
            try:
                print("尝试使用stock_zh_a_hist接口...")
                df = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily",
                                        start_date=self.start_date,
                                        end_date=self.end_date, adjust="hfq")
                # 兼容不同版本的列名
                column_mapping = {
                    '日期': 'date', 'date': 'date',
                    '开盘': 'open', 'open': 'open',
                    '最高': 'high', 'high': 'high',
                    '最低': 'low', 'low': 'low',
                    '收盘': 'close', 'close': 'close',
                    '成交量': 'volume', 'volume': 'volume',
                    '成交额': 'amount', 'amount': 'amount'
                }
                # 重命名列
                df = df.rename(columns={col: column_mapping[col] for col in df.columns if col in column_mapping})
            except Exception as e2:
                print(f"无法获取股票数据: {e2}")
                return pd.DataFrame()

        # 确保必要的列存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"数据缺少必要列: {missing}")
            return pd.DataFrame()

        # 转换日期格式并排序
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=True)
        df.set_index('date', inplace=True)

        # 过滤无效数据
        df = df[df['volume'] > 0]
        return df

    def _calculate_indicators(self):
        """计算技术指标（纯Pandas实现）"""
        # 基础均线系统
        ma_periods = [5, 10, 20, 60, 120, 250]
        for period in ma_periods:
            self.data[f'MA{period}'] = self.data['close'].rolling(window=period).mean()

        # 成交量分析
        self.data['VOL_MA5'] = self.data['volume'].rolling(window=5).mean()
        self.data['VOL_MA10'] = self.data['volume'].rolling(window=10).mean()
        self.data['VOL_RATIO'] = self.data['volume'] / self.data['VOL_MA5']  # 量比

        # MACD指标 (纯Pandas实现)
        self.data['MACD'], self.data['MACD_SIGNAL'], self.data['MACD_HIST'] = pandas_MACD(self.data['close'])

        # KDJ指标 (带除零保护)
        self.data['K'], self.data['D'], self.data['J'] = pandas_KDJ(
            self.data['high'], self.data['low'], self.data['close'])

        # RSI指标 (纯Pandas实现)
        self.data['RSI6'] = pandas_RSI(self.data['close'], period=6)
        self.data['RSI12'] = pandas_RSI(self.data['close'], period=12)
        self.data['RSI24'] = pandas_RSI(self.data['close'], period=24)

        # 布林带 (纯Pandas实现)
        self.data['BOLL_UPPER'], self.data['BOLL_MID'], self.data['BOLL_LOWER'] = pandas_BOLL(self.data['close'])

        # 动态支撑/压力线
        self.data['UPPER'] = self.data['high'].rolling(20).max()
        self.data['LOWER'] = self.data['low'].rolling(20).min()
        self.data['DYNAMIC_SUPPORT'] = (self.data['MA20'] + self.data['LOWER']) / 2
        self.data['DYNAMIC_RESIST'] = (self.data['MA20'] + self.data['UPPER']) / 2

        # 主力资金指标
        self.data['PRICE_CHANGE'] = self.data['close'].pct_change() * 100
        self.data['AMOUNT_MA5'] = self.data['amount'].rolling(5).mean()

        # 九品神龙信号
        self._generate_signals()

    def _generate_signals(self):
        """生成买卖信号"""
        # 1. 均线系统信号
        # 多头排列条件
        ma_condition = (self.data['MA5'] > self.data['MA10']) & \
                       (self.data['MA10'] > self.data['MA20']) & \
                       (self.data['MA20'] > self.data['MA60'])

        # 均线金叉
        self.data['MA5_MA10_CROSS'] = (self.data['MA5'] > self.data['MA10']) & (
                self.data['MA5'].shift(1) <= self.data['MA10'].shift(1))
        self.data['MA10_MA20_CROSS'] = (self.data['MA10'] > self.data['MA20']) & (
                self.data['MA10'].shift(1) <= self.data['MA20'].shift(1))

        # 2. 量能信号
        vol_condition = (self.data['volume'] > self.data['VOL_MA5'] * 1.5) & \
                        (self.data['volume'] > self.data['VOL_MA10'] * 1.2)

        # 3. MACD信号
        macd_condition = (self.data['MACD'] > self.data['MACD_SIGNAL']) & \
                         (self.data['MACD_HIST'] > 0) & \
                         (self.data['MACD_HIST'] > self.data['MACD_HIST'].shift(1))

        # 4. KDJ信号
        kdj_condition = (self.data['K'] > self.data['D']) & (self.data['K'] < 80) & (self.data['J'] > 0)

        # 5. 趋势突破信号
        trend_condition = (self.data['close'] > self.data['DYNAMIC_RESIST']) & \
                          (self.data['close'] > self.data['BOLL_MID'])

        # 综合买入信号 (九品神龙核心信号)
        self.data['BUY_SIGNAL'] = ma_condition & vol_condition & macd_condition & kdj_condition & trend_condition

        # 卖出信号
        sell_condition1 = self.data['close'] < self.data['MA20']  # 跌破20日均线
        sell_condition2 = (self.data['MACD'] < self.data['MACD_SIGNAL']) & (self.data['MACD_HIST'] < 0)  # MACD死叉
        sell_condition3 = self.data['J'] < 0  # J线进入超卖区
        self.data['SELL_SIGNAL'] = sell_condition1 | sell_condition2 | sell_condition3

        # 仓位管理信号 (向量化操作)
        # 初始化仓位为0
        self.data['POSITION'] = 0

        # 买入信号位置
        buy_signals = self.data[self.data['BUY_SIGNAL']].index

        # 卖出信号位置
        sell_signals = self.data[self.data['SELL_SIGNAL']].index

        # 创建仓位序列
        position_series = pd.Series(0, index=self.data.index)

        # 处理买入信号
        for buy_date in buy_signals:
            # 找到下一个卖出信号
            next_sell = sell_signals[sell_signals > buy_date].min()
            if pd.notnull(next_sell):
                # 从买入日到卖出日前一日设为持仓
                position_series.loc[buy_date:next_sell - pd.Timedelta(days=1)] = 1
            else:
                # 如果没有后续卖出信号，则一直持仓
                position_series.loc[buy_date:] = 1

        # 设置仓位数据
        self.data['POSITION'] = position_series

    def plot_analysis(self, last_days=120):
        """绘制高级分析图表（优化布局）"""
        if self.data.empty:
            print("没有数据可供绘图")
            return

        plot_data = self.data.last(f'{last_days}D')

        # 创建图表和子图
        fig = plt.figure(figsize=(18, 16), dpi=100)
        gs = fig.add_gridspec(6, 1, height_ratios=[3, 1, 1, 1, 1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        ax6 = fig.add_subplot(gs[5], sharex=ax1)

        # 使用mplfinance绘制K线图
        apds = [
            # 添加均线
            mpf.make_addplot(plot_data['MA5'], ax=ax1, color='blue', width=1, panel=0, ylabel='价格'),
            mpf.make_addplot(plot_data['MA10'], ax=ax1, color='orange', width=1, panel=0),
            mpf.make_addplot(plot_data['MA20'], ax=ax1, color='green', width=1, panel=0),
            mpf.make_addplot(plot_data['MA60'], ax=ax1, color='purple', width=1.5, panel=0),

            # 添加布林带
            mpf.make_addplot(plot_data['BOLL_UPPER'], ax=ax1, color='red', linestyle='--', alpha=0.7, panel=0),
            mpf.make_addplot(plot_data['BOLL_MID'], ax=ax1, color='blue', alpha=0.7, panel=0),
            mpf.make_addplot(plot_data['BOLL_LOWER'], ax=ax1, color='red', linestyle='--', alpha=0.7, panel=0),
        ]

        # 绘制K线图
        mpf.plot(plot_data, type='candle', style='charles', ax=ax1, addplot=apds,
                 show_nontrading=False, warn_too_much_data=len(plot_data) + 1)

        # 标记买卖信号
        buy_signals = plot_data[plot_data['BUY_SIGNAL']]
        sell_signals = plot_data[plot_data['SELL_SIGNAL']]
        ax1.scatter(buy_signals.index, buy_signals['low'] * 0.98,
                    marker='^', color='red', s=100, label='买入信号')
        ax1.scatter(sell_signals.index, sell_signals['high'] * 1.02,
                    marker='v', color='green', s=100, label='卖出信号')

        # 标记仓位
        position_changes = plot_data[plot_data['POSITION'].diff() != 0]
        for idx, row in position_changes.iterrows():
            if row['POSITION'] == 1:
                ax1.axvline(x=idx, color='blue', linestyle='-', alpha=0.3)
            else:
                ax1.axvline(x=idx, color='gray', linestyle='-', alpha=0.3)

        ax1.set_title(f'{self.stock_code} 九品神龙分析', fontsize=16)
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 成交量 - 修复颜色问题
        # 生成颜色列表：上涨日红色，下跌日绿色
        colors = np.where(plot_data['close'] >= plot_data['open'], 'red', 'green')
        ax2.bar(plot_data.index, plot_data['volume'], color=colors)
        ax2.plot(plot_data['VOL_MA5'], label='5日成交量均线', color='blue')
        ax2.plot(plot_data['VOL_MA10'], label='10日成交量均线', color='orange')
        ax2.set_ylabel('成交量')
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # MACD
        ax3.plot(plot_data['MACD'], label='MACD', color='blue')
        ax3.plot(plot_data['MACD_SIGNAL'], label='Signal', color='red')
        # 绘制MACD柱状图
        ax3.bar(plot_data.index, plot_data['MACD_HIST'],
                color=np.where(plot_data['MACD_HIST'] >= 0, 'red', 'green'), alpha=0.7)
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_ylabel('MACD')
        ax3.legend(loc='best')
        ax3.grid(True, linestyle='--', alpha=0.7)

        # KDJ
        ax4.plot(plot_data['K'], label='K线', color='blue')
        ax4.plot(plot_data['D'], label='D线', color='red')
        ax4.plot(plot_data['J'], label='J线', color='green')
        ax4.axhline(80, color='gray', linestyle='--', alpha=0.5)
        ax4.axhline(20, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylabel('KDJ')
        ax4.legend(loc='best')
        ax4.grid(True, linestyle='--', alpha=0.7)

        # RSI
        ax5.plot(plot_data['RSI6'], label='RSI6', color='blue')
        ax5.plot(plot_data['RSI12'], label='RSI12', color='red')
        ax5.plot(plot_data['RSI24'], label='RSI24', color='green')
        ax5.axhline(70, color='gray', linestyle='--', alpha=0.5)
        ax5.axhline(30, color='gray', linestyle='--', alpha=0.5)
        ax5.set_ylabel('RSI')
        ax5.legend(loc='best')
        ax5.grid(True, linestyle='--', alpha=0.7)

        # 资金流向
        ax6.plot(plot_data['PRICE_CHANGE'], label='价格涨跌%', color='blue')
        ax6.bar(plot_data.index, plot_data['amount'] / 1000000, color='orange', alpha=0.5, label='成交额(百万)')
        ax6.set_ylabel('资金流向')
        ax6.legend(loc='best')
        ax6.grid(True, linestyle='--', alpha=0.7)

        # 调整布局
        plt.subplots_adjust(hspace=0.1)
        plt.tight_layout()
        plt.show()

    def generate_signals_report(self):
        """生成信号报告"""
        if self.data.empty or len(self.data) < 60:
            return pd.DataFrame({"错误": ["数据不足，至少需要60个交易日数据"]})

        latest = self.data.iloc[-1]
        report = {
            "股票代码": self.stock_code,
            "当前价格": f"{latest['close']:.2f}",
            "短期趋势": "多头" if latest['close'] > latest['MA5'] > latest['MA10'] else "空头",
            "中期趋势": "多头" if latest['close'] > latest['MA20'] > latest['MA60'] else "空头",
            "长期趋势": "多头" if latest['close'] > latest['MA120'] > latest['MA250'] else "空头",
            "均线排列": self._get_ma_arrangement(latest),
            "量能状态": "放量" if latest['volume'] > latest['VOL_MA5'] * 1.5 else "缩量",
            "MACD状态": "金叉" if latest['MACD'] > latest['MACD_SIGNAL'] else "死叉",
            "KDJ状态": f"K:{latest['K']:.1f}, D:{latest['D']:.1f}, J:{latest['J']:.1f}",
            "RSI状态": f"RSI6:{latest['RSI6']:.1f}, RSI12:{latest['RSI12']:.1f}",
            "支撑位": f"{latest['DYNAMIC_SUPPORT']:.2f} (动态), {latest['BOLL_LOWER']:.2f} (布林)",
            "压力位": f"{latest['DYNAMIC_RESIST']:.2f} (动态), {latest['BOLL_UPPER']:.2f} (布林)",
            "主力动向": "流入" if latest['PRICE_CHANGE'] > 0 and latest['volume'] > latest['VOL_MA5'] else "流出",
            "当前信号": "买入" if latest['BUY_SIGNAL'] else "卖出" if latest['SELL_SIGNAL'] else "观望",
            "操作建议": self._get_trading_suggestion(latest)
        }
        return pd.DataFrame(report.items(), columns=['指标', '值'])

    def _get_ma_arrangement(self, data_point):
        """判断均线排列状态"""
        if data_point['MA5'] > data_point['MA10'] > data_point['MA20'] > data_point['MA60'] > data_point['MA120'] > \
                data_point['MA250']:
            return "完美多头"
        elif data_point['MA5'] < data_point['MA10'] < data_point['MA20'] < data_point['MA60'] < data_point['MA120'] < \
                data_point['MA250']:
            return "空头排列"
        elif data_point['close'] > data_point['MA5'] > data_point['MA10']:
            return "短期多头"
        elif data_point['close'] > data_point['MA20'] > data_point['MA60']:
            return "中期多头"
        else:
            return "震荡排列"

    def _get_trading_suggestion(self, data_point):
        """生成交易建议"""
        if data_point['BUY_SIGNAL']:
            return "强烈买入信号，可建仓或加仓"
        elif data_point['SELL_SIGNAL']:
            return "强烈卖出信号，应减仓或清仓"
        elif data_point['close'] > data_point['MA20']:
            if data_point['volume'] > data_point['VOL_MA5'] * 1.2:
                return "量价齐升，可持有观望"
            else:
                return "趋势向上但量能不足，谨慎持有"
        else:
            return "空仓等待更好机会"

    def backtest_strategy(self, initial_capital=100000):
        """策略回测（向量化操作优化）"""
        if self.data.empty or len(self.data) < 100:
            print("数据不足，无法进行有效回测")
            return None

        # 创建回测数据副本
        df = self.data.copy()

        # 初始化回测参数
        df['Position'] = df['POSITION']
        df['Shares'] = 0
        df['Cash'] = initial_capital
        df['Total'] = initial_capital

        # 向量化操作计算交易
        # 找到所有买入和卖出点
        buy_points = df[df['Position'].diff() == 1].index
        sell_points = df[df['Position'].diff() == -1].index

        # 处理初始状态
        current_cash = initial_capital
        current_shares = 0

        # 记录交易
        trades = []

        # 处理买入点
        for date in buy_points:
            price = df.loc[date, 'close']
            # 计算可买股数（整数）
            shares = current_cash // price
            if shares > 0:
                cost = shares * price
                current_cash -= cost
                current_shares = shares
                trades.append({
                    'date': date,
                    'type': 'buy',
                    'price': price,
                    'shares': shares,
                    'value': cost
                })

        # 处理卖出点
        for date in sell_points:
            if current_shares > 0:
                price = df.loc[date, 'close']
                value = current_shares * price
                current_cash += value
                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': price,
                    'shares': current_shares,
                    'value': value
                })
                current_shares = 0

        # 更新每日持仓价值
        df['Shares'] = current_shares
        df['PositionValue'] = df['Shares'] * df['close']
        df['Cash'] = current_cash
        df['Total'] = df['Cash'] + df['PositionValue']

        # 计算绩效指标
        final_value = df['Total'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital * 100

        # 最大回撤
        df['Peak'] = df['Total'].cummax()
        df['Drawdown'] = (df['Total'] - df['Peak']) / df['Peak']
        max_drawdown = df['Drawdown'].min() * 100

        # 交易次数
        num_trades = len(trades) // 2 if len(trades) % 2 == 0 else (len(trades) - 1) // 2

        # 胜率
        win_rate = 0
        if num_trades > 0:
            wins = 0
            for i in range(0, len(trades), 2):
                if i + 1 < len(trades):
                    buy_trade = trades[i]
                    sell_trade = trades[i + 1]
                    if sell_trade['price'] > buy_trade['price']:
                        wins += 1
            win_rate = wins / num_trades * 100

        # 年化收益率
        days = (df.index[-1] - df.index[0]).days
        annualized_return = 0
        if days > 0:
            annualized_return = ((final_value / initial_capital) ** (365.25 / days) - 1) * 100

        # 生成回测报告
        report = {
            "初始资金": f"{initial_capital:,.2f}元",
            "最终资产": f"{final_value:,.2f}元",
            "总收益率": f"{total_return:.2f}%",
            "年化收益率": f"{annualized_return:.2f}%",
            "交易次数": num_trades,
            "胜率": f"{win_rate:.1f}%" if num_trades > 0 else "无交易",
            "最大回撤": f"{max_drawdown:.2f}%",
            "持仓天数": days,
            "最终仓位": "持仓" if current_shares > 0 else "空仓"
        }

        return pd.DataFrame(report.items(), columns=['指标', '值'])

    @staticmethod
    def get_low_open_leaders(top_n=10, exclude_tech=True, exclude_gem=True, exclude_bj=True, min_amount=100000000):
        """
        获取低开的前N个龙头股，增加过滤条件
        :param top_n: 返回的股票数量
        :param exclude_tech: 是否排除科技股
        :param exclude_gem: 是否排除创业板股票
        :param exclude_bj: 是否排除北京交易所股票
        :param min_amount: 最小交易额(元)，默认为1亿元
        :return: 符合条件的股票DataFrame
        """
        try:
            print("正在获取所有A股实时行情...")
            # 使用东方财富接口替代新浪接口
            spot_df = ak.stock_zh_a_spot_em()  # 使用东方财富的接口

            # 重命名列以统一格式
            spot_df = spot_df.rename(columns={
                '代码': '代码',
                '名称': '名称',
                '最新价': '最新价',
                '涨跌幅': '涨跌幅',
                '涨跌额': '涨跌额',
                '成交量': '成交量',
                '成交额': '成交额',
                '今开': '今开',
                '昨收': '昨收',
                '最高': '最高',
                '最低': '最低'
            })

            # 确保代码列是字符串类型
            spot_df['代码'] = spot_df['代码'].astype(str)

            # 检查关键列是否存在
            required_columns = ['代码', '名称', '今开', '昨收', '成交量', '成交额', '最新价']
            missing_cols = [col for col in required_columns if col not in spot_df.columns]
            if missing_cols:
                print(f"缺少必要列: {missing_cols}")
                return pd.DataFrame()

            # 过滤无效数据 - 确保只分析可交易的股票
            # 1. 排除开盘价为0的股票（停牌或退市）
            spot_df = spot_df[spot_df['今开'] > 0]
            # 2. 排除昨日收盘价为0的股票
            spot_df = spot_df[spot_df['昨收'] > 0]
            # 3. 排除ST/*ST股票
            spot_df = spot_df[~spot_df['名称'].str.contains('ST')]
            # 4. 排除北京交易所股票
            if exclude_bj:
                # 匹配以'8'开头或以'bj'开头的代码
                bj_mask = spot_df['代码'].str.startswith('8') | spot_df['代码'].str.startswith('bj')
                spot_df = spot_df[~bj_mask]
                print(f"排除北京交易所股票后，剩余{len(spot_df)}只股票")

            # 排除创业板股票
            if exclude_gem:
                # 匹配以'300'或'301'开头的创业板股票
                gem_mask = spot_df['代码'].str.startswith('300') | spot_df['代码'].str.startswith('301')
                spot_df = spot_df[~gem_mask]
                print(f"排除创业板股票后，剩余{len(spot_df)}只股票")

            # 排除科技股（根据行业分类）
            if exclude_tech:
                # 定义科技行业关键词
                tech_keywords = ['科技', '软件', '通信', '电子', '互联网', '计算机', '半导体', '信息', '芯片', '网络']

                # 标记科技股
                spot_df['is_tech'] = False
                for keyword in tech_keywords:
                    spot_df.loc[spot_df['名称'].str.contains(keyword, na=False), 'is_tech'] = True

                # 排除科技股
                original_count = len(spot_df)
                spot_df = spot_df[~spot_df['is_tech']]
                print(f"排除科技股后，剩余{len(spot_df)}只股票（原{original_count}只）")

            # 筛选交易额达到亿元级别的股票
            if '成交额' in spot_df.columns:
                # 转换成交额为数值类型（如果需要）
                if spot_df['成交额'].dtype == 'object':
                    try:
                        # 处理可能的单位问题（如亿元、万元等）
                        if any('亿' in str(x) for x in spot_df['成交额'].head(3)):
                            spot_df['成交额'] = spot_df['成交额'].str.replace('亿', '').astype(float) * 1e8
                        elif any('万' in str(x) for x in spot_df['成交额'].head(3)):
                            spot_df['成交额'] = spot_df['成交额'].str.replace('万', '').astype(float) * 1e4
                    except Exception as e:
                        print(f"成交额转换失败: {e}, 使用原始数据")
                        try:
                            spot_df['成交额'] = spot_df['成交额'].astype(float)
                        except:
                            print("无法转换成交额为数值类型")
                else:
                    # 确保是数值类型
                    spot_df['成交额'] = spot_df['成交额'].astype(float)

                # 筛选交易额大于等于min_amount的股票
                original_count = len(spot_df)
                spot_df = spot_df[spot_df['成交额'] >= min_amount]
                print(f"筛选交易额≥{min_amount / 1e8:.2f}亿元的股票后，剩余{len(spot_df)}只股票（原{original_count}只）")
            else:
                print("警告: 未找到成交额列，无法进行交易额筛选")
                return pd.DataFrame()

            if spot_df.empty:
                print("筛选后无符合条件的股票")
                return pd.DataFrame()

            # 计算涨跌幅
            spot_df['开盘相对昨收涨跌幅'] = (spot_df['今开'] - spot_df['昨收']) / spot_df['昨收'] * 100
            spot_df['当日总涨跌幅'] = (spot_df['最新价'] - spot_df['昨收']) / spot_df['昨收'] * 100
            spot_df['开盘至今涨跌幅'] = (spot_df['最新价'] - spot_df['今开']) / spot_df['今开'] * 100

            # 筛选低开股票（开盘价低于昨日收盘价）
            low_open_df = spot_df[spot_df['开盘相对昨收涨跌幅'] < 0]

            if low_open_df.empty:
                print("未找到低开股票")
                return pd.DataFrame()

            # 按开盘跌幅排序（跌幅最大的在前）
            low_open_df = low_open_df.sort_values(by='开盘相对昨收涨跌幅')

            # 取前N个
            top_low_open = low_open_df.head(top_n)

            # 添加行业信息
            print("正在获取行业信息...")
            industry_map = {}
            for idx, row in top_low_open.iterrows():
                try:
                    code = row['代码']
                    # 尝试获取行业信息
                    industry_info = ak.stock_individual_info_em(symbol=code)
                    if not industry_info.empty:
                        # 查找行业信息
                        industry_row = industry_info[industry_info['item'] == '所属行业']
                        if not industry_row.empty:
                            industry = industry_row.iloc[0]['value']
                            industry_map[code] = industry
                        else:
                            industry_map[code] = "未知"
                    else:
                        industry_map[code] = "未知"
                    # 添加延迟避免请求过快
                    time.sleep(0.5)
                except Exception as e:
                    print(f"获取股票 {code} 行业信息失败: {e}")
                    industry_map[code] = "未知"

            # 添加行业列
            top_low_open['行业'] = top_low_open['代码'].map(industry_map)

            # 格式化数据
            top_low_open['成交额(亿)'] = top_low_open['成交额'] / 1e8

            # 选择需要的列
            result = top_low_open[['代码', '名称', '最新价', '今开', '昨收',
                                   '开盘相对昨收涨跌幅', '当日总涨跌幅', '开盘至今涨跌幅', '成交额(亿)', '行业']]

            # 四舍五入小数位数
            result = result.round({
                '最新价': 2,
                '今开': 2,
                '昨收': 2,
                '开盘相对昨收涨跌幅': 2,
                '当日总涨跌幅': 2,
                '开盘至今涨跌幅': 2,
                '成交额(亿)': 2
            })

            return result

        except Exception as e:
            print(f"获取龙头股失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    @staticmethod
    def generate_buy_recommendation(stocks_df):
        """
        生成买入推荐策略
        :param stocks_df: 低开龙头股DataFrame
        :return: 买入推荐DataFrame和组合预期收益
        """
        if stocks_df.empty:
            return pd.DataFrame(), 0.0

        # 复制数据避免修改原始数据
        df = stocks_df.copy()

        # 1. 筛选条件：价格在5-50元之间
        df = df[(df['今开'] >= 5) & (df['今开'] <= 50)]

        # 2. 按低开幅度排序（跌幅最大的优先）
        df = df.sort_values(by='开盘相对昨收涨跌幅')

        # 3. 计算买入评分（综合考虑低开幅度、成交额和价格）
        df['低开幅度得分'] = df['开盘相对昨收涨跌幅'].rank(ascending=True) * 40
        df['成交额得分'] = df['成交额(亿)'].rank(ascending=False) * 30
        df['价格稳定性得分'] = (1 / (df['今开'] - df['昨收']).abs().rank(ascending=True)) * 30
        df['综合得分'] = df['低开幅度得分'] + df['成交额得分'] + df['价格稳定性得分']

        # 4. 选择前3只股票
        top3 = df.head(3).copy()

        if len(top3) == 0:
            return pd.DataFrame(), 0.0

        # 5. 分配仓位
        positions = ['40%', '30%', '20%'][:len(top3)]
        top3['建议仓位'] = positions

        # 6. 设置止损价（-3%）
        top3['止损价'] = top3['今开'] * 0.97

        # 7. 设置止盈目标
        top3['第一目标价'] = top3['今开'] * 1.05
        top3['第二目标价'] = top3['今开'] * 1.10

        # 8. 计算预期收益
        top3['预期最低收益'] = top3['第一目标价'] / top3['今开'] - 1
        top3['预期最高收益'] = top3['第二目标价'] / top3['今开'] - 1

        # 9. 添加推荐理由
        reasons = []
        for idx, row in top3.iterrows():
            reason = f"低开幅度大({row['开盘相对昨收涨跌幅']:.2f}%)，成交活跃({row['成交额(亿)']:.2f}亿)"
            reasons.append(reason)
        top3['推荐理由'] = reasons

        # 10. 计算组合预期收益
        weighted_return = 0
        weights = [0.4, 0.3, 0.2][:len(top3)]

        for i, (_, row) in enumerate(top3.iterrows()):
            weight = weights[i]
            weighted_return += weight * (row['预期最低收益'] + row['预期最高收益']) / 2

        # 返回推荐结果和组合预期
        return top3[['代码', '名称', '今开', '止损价', '第一目标价', '第二目标价',
                     '预期最低收益', '预期最高收益', '建议仓位', '推荐理由']], weighted_return


# 使用示例
if __name__ == "__main__":
    # 自动获取符合条件的股票
    print("\n" + "=" * 70)
    print("九品神龙股票分析系统 - 龙头股筛选与智能买入策略")
    print("=" * 70)

    # 用户可配置参数
    top_n = 10
    min_turnover = 100000000  # 1亿元

    # 筛选条件：排除科技股、创业板股票、北京交易所股票，交易额≥1亿元
    qualified_stocks = JiuPinShenLong.get_low_open_leaders(
        top_n=top_n,
        exclude_tech=True,
        exclude_gem=True,
        exclude_bj=True,
        min_amount=min_turnover
    )

    if not qualified_stocks.empty:
        # 格式化输出股票信息
        print("\n低开龙头股排行榜:")
        # 添加表头
        header = (f"{'序号':<4}{'代码':<8}{'名称':<10}{'最新价':<8}{'今开':<8}{'昨收':<8}"
                  f"{'开盘相对昨收':<12}{'当日总涨跌':<12}{'开盘至今涨跌':<12}"
                  f"{'成交额(亿)':<12}{'行业':<20}")

        header_length = len(header) + 10
        print("-" * header_length)
        print(header)
        print("-" * header_length)

        # 格式化输出每行数据
        for i, (idx, row) in enumerate(qualified_stocks.iterrows(), 1):
            # 获取数据
            code = row['代码']
            name = row['名称']
            current_price = row['最新价']
            open_price = row['今开']
            prev_close = row['昨收']

            # 获取涨跌幅数据
            open_to_prev = row['开盘相对昨收涨跌幅']
            daily_change = row['当日总涨跌幅']
            intraday_change = row['开盘至今涨跌幅']

            # 格式化成交额
            turnover = row['成交额(亿)']

            # 获取行业
            industry = row.get('行业', '未知')


            # 格式化涨跌幅字符串（带+/-号）
            def format_change(change):
                sign = '+' if change >= 0 else ''
                return f"{sign}{change:.2f}%"


            open_to_prev_str = format_change(open_to_prev)
            daily_change_str = format_change(daily_change)
            intraday_change_str = format_change(intraday_change)

            # 构建行字符串
            row_str = (f"{i:<4}{code:<8}{name:<10}{current_price:<8.2f}{open_price:<8.2f}{prev_close:<8.2f}"
                       f"{open_to_prev_str:<12}{daily_change_str:<12}{intraday_change_str:<12}"
                       f"{turnover:<12.2f}{industry:<20}")
            print(row_str)

        print("-" * header_length)

        # 添加解释说明
        print("\n说明:")
        print("1. 开盘相对昨收: 开盘价相对于昨日收盘价的涨跌幅")
        print("2. 当日总涨跌: 当前价相对于昨日收盘价的涨跌幅")
        print("3. 开盘至今涨跌: 当前价相对于今日开盘价的涨跌幅")

        # 生成买入推荐
        print("\n" + "=" * 70)
        print("智能买入策略推荐 (基于开盘数据分析)")
        print("=" * 70)

        buy_recommendation, portfolio_return = JiuPinShenLong.generate_buy_recommendation(qualified_stocks)

        if not buy_recommendation.empty:
            # 输出买入推荐
            buy_header = (f"{'序号':<4}{'代码':<8}{'名称':<10}{'开盘价':<8}{'止损价':<8}{'第一目标':<8}{'第二目标':<8}"
                          f"{'最低收益':<10}{'最高收益':<10}{'仓位':<8}{'推荐理由':<30}")

            buy_header_length = len(buy_header) + 10
            print("-" * buy_header_length)
            print(buy_header)
            print("-" * buy_header_length)

            for i, (idx, row) in enumerate(buy_recommendation.iterrows(), 1):
                code = row['代码']
                name = row['名称']
                open_price = row['今开']
                stop_loss = row['止损价']
                target1 = row['第一目标价']
                target2 = row['第二目标价']
                min_return = row['预期最低收益'] * 100
                max_return = row['预期最高收益'] * 100
                position = row['建议仓位']
                reason = row['推荐理由']

                # 构建行字符串
                buy_row = (f"{i:<4}{code:<8}{name:<10}{open_price:<8.2f}{stop_loss:<8.2f}{target1:<8.2f}{target2:<8.2f}"
                           f"{min_return:<10.2f}%{max_return:<10.2f}%{position:<8}{reason:<30}")
                print(buy_row)

            print("-" * buy_header_length)

            # 输出组合预期
            print(f"\n组合预期收益: {portfolio_return * 100:.2f}%")
            print("仓位分配策略: 40%(首选) + 30%(次选) + 20%(三选) + 10%(现金)")
            print("风险控制: 所有股票设置3%止损位")

            # 输出投资建议
            print("\n投资建议:")
            print("1. 优先选择低开幅度大且成交活跃的股票")
            print("2. 开盘后30分钟内分批建仓")
            print("3. 触及止损价坚决止损")
            print("4. 达到第一目标价可部分止盈，保留剩余仓位看第二目标")
            print("5. 保留10%现金应对市场波动")
        else:
            print("未找到符合条件的买入标的")
    else:
        print("未找到符合条件的股票")