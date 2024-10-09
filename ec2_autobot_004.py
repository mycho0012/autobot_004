import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyupbit
import requests
import logging
import pandas_ta as ta
import uuid  # Trade ID 생성을 위해 사용
from dotenv import load_dotenv

# 환경 변수 로드 (권장: .env 파일 사용)
load_dotenv()

# 설정
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
NOTION_API_TOKEN = os.getenv('NOTION_API_TOKEN')
NOTION_DATABASE_ID = os.getenv('NOTION_DATABASE_ID')

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

def send_telegram_message(message: str):
    """텔레그램 봇을 통해 메시지를 전송합니다."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message
        }
        response = requests.post(url, data=data)
        if response.status_code != 200:
            logger.error(f"Failed to send Telegram message: {response.text}")
        else:
            logger.info("Telegram 메시지 전송 완료")
    except Exception as e:
        logger.error(f"Exception in send_telegram_message: {e}")

def log_to_notion(trade: dict):
    """Notion 데이터베이스에 거래 내역을 기록합니다."""
    try:
        headers = {
            "Authorization": f"Bearer {NOTION_API_TOKEN}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        properties = {
            "Trade ID": {"title": [{"text": {"content": str(trade['Trade ID'])}}]},
            "Ticker": {"rich_text": [{"text": {"content": trade['Ticker']}}]},
            "Entry Time": {"date": {"start": trade['Entry Time'].isoformat()}},
            "Exit Time": {"date": {"start": trade['Exit Time'].isoformat()} if trade['Exit Time'] else None},
            "Signal Type": {"rich_text": [{"text": {"content": trade['Signal Type']}}]},
            "Entry Price": {"number": trade['Entry Price']},
            "Exit Price": {"number": trade['Exit Price'] if trade['Exit Price'] is not None else None},
            "Log Close Price": {"number": trade['Log Close Price']},
            "Resistance Level": {"number": trade['Resistance Level']},
            "Support Level": {"number": trade['Support Level']},
            "Volume": {"number": trade['Volume']},
            "ATR": {"number": trade['ATR']},
            "Reason for Entry": {"rich_text": [{"text": {"content": trade['Reason for Entry']}}]},
            "Reason for Exit": {"rich_text": [{"text": {"content": trade['Reason for Exit']}}] if trade['Reason for Exit'] else None},
            "Profit/Loss": {"number": trade['Profit/Loss'] if trade['Profit/Loss'] is not None else None},
            "Return (%)": {"number": trade['Return (%)'] if trade['Return (%)'] is not None else None},
        }
        data = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": properties
        }
        response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"Notion 데이터베이스 업데이트 실패: {response.text}")
        else:
            logger.info("Notion 데이터베이스 업데이트 완료")
    except Exception as e:
        logger.error(f"Exception in log_to_notion: {e}")

class MRHATradingSystem:
    def __init__(self, symbol, interval, count, atr_period=14):
        self.symbol = symbol
        self.interval = interval
        self.count = count
        self.atr_period = atr_period
        self.stock_data = None
        self.mrha_data = None

    def download_data(self):
        df = pyupbit.get_ohlcv(self.symbol, interval=self.interval, count=self.count)
        if df is None or df.empty:
            raise ValueError("데이터 다운로드 실패 또는 빈 데이터 프레임입니다.")
        df = df.rename(columns=lambda x: x.capitalize())
        df = df.drop(columns='Value', errors='ignore')
        df.index.name = 'Date'
        self.stock_data = df
        return self.stock_data

    def calculate_revised_heikin_ashi(self):
        if self.stock_data.index.duplicated().any():
            raise ValueError("stock_data 인덱스에 중복 날짜가 있습니다. 데이터를 확인하세요.")

        ha = self.stock_data[['Open', 'High', 'Low', 'Close']].copy()
        ha.columns = ['h_open', 'h_high', 'h_low', 'h_close']
        ha['h_close'] = (ha['h_open'] + ha['h_high'] + ha['h_low'] + ha['h_close']) / 4
        for i in range(1, len(ha)):
            ha.iloc[i, 0] = (ha.iloc[i-1, 0] + ha.iloc[i-1, 3]) / 2
        ha['h_high'] = ha[['h_open', 'h_close']].join(self.stock_data['High']).max(axis=1)
        ha['h_low'] = ha[['h_open', 'h_close']].join(self.stock_data['Low']).min(axis=1)
        return ha

    def calculate_mrha(self, rha_data):
        mrha = pd.DataFrame(index=self.stock_data.index, columns=['mh_open', 'mh_high', 'mh_low', 'mh_close'])
        mrha['mh_open'] = (rha_data['h_open'] + rha_data['h_close']) / 2
        mrha['mh_high'] = rha_data['h_open'].rolling(window=5).mean()
        mrha['mh_low'] = rha_data['h_low'].rolling(window=5).mean()
        mrha['mh_close'] = (mrha['mh_open'] + self.stock_data['High'] + self.stock_data['Low'] + self.stock_data['Close'] * 2) / 5
        return mrha.dropna()

    def add_trading_signals(self):
        def calculate_ebr(mh_open, low):
            return (4 * mh_open - low) / 3

        def calculate_btrg(ebr):
            return 1.00618 * ebr

        def calculate_ebl(mh_open, high):
            return (4 * mh_open - high) / 3

        def calculate_strg(ebl):
            return 0.99382 * ebl

        signals = pd.DataFrame(index=self.mrha_data.index)
        signals['Ebr'] = calculate_ebr(self.mrha_data['mh_open'], self.stock_data['Low'])
        signals['Btrg'] = calculate_btrg(signals['Ebr'])
        signals['Ebl'] = calculate_ebl(self.mrha_data['mh_open'], self.stock_data['High'])
        signals['Strg'] = calculate_strg(signals['Ebl'])

        # ATR 계산 (pandas_ta 사용)
        atr = ta.atr(high=self.stock_data['High'], low=self.stock_data['Low'], close=self.stock_data['Close'], length=self.atr_period)
        signals['ATR'] = atr

        # Stop Loss 및 Take Profit 레벨 설정 (ATR 기반)
        signals['Stop_Loss'] = self.mrha_data['mh_close'] - (signals['ATR'] * 1.5)
        signals['Take_Profit'] = self.mrha_data['mh_close'] + (signals['ATR'] * 3)

        self.mrha_data = pd.concat([self.mrha_data, signals], axis=1)

    def implement_trading_logic(self):
        signals = pd.DataFrame(index=self.mrha_data.index, columns=['Signal'])
        position = 0

        for i in range(1, len(self.mrha_data)):
            bullish_candle = self.mrha_data['mh_close'].iloc[i] > self.mrha_data['mh_open'].iloc[i] and \
                             self.mrha_data['mh_close'].iloc[i] > self.mrha_data['mh_high'].iloc[i-1]

            if position == 0 and bullish_candle and self.mrha_data['mh_close'].iloc[i] > self.mrha_data['Btrg'].iloc[i]:
                signals['Signal'].iloc[i] = 1
                position = 1
            elif position == 1 and (self.mrha_data['mh_close'].iloc[i] < self.mrha_data['Stop_Loss'].iloc[i] or 
                                    self.mrha_data['mh_close'].iloc[i] > self.mrha_data['Take_Profit'].iloc[i]):
                signals['Signal'].iloc[i] = -1
                position = 0
            else:
                signals['Signal'].iloc[i] = 0

        self.mrha_data = pd.concat([self.mrha_data, signals], axis=1)

    def run_analysis(self):
        self.download_data()
        rha_data = self.calculate_revised_heikin_ashi()
        self.mrha_data = self.calculate_mrha(rha_data)
        self.add_trading_signals()
        self.implement_trading_logic()

class TradingBot:
    def __init__(self):
        self.trading_system = MRHATradingSystem(symbol="KRW-BTC", interval="day", count=200)
        self.upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.current_trade = None  # 현재 진행 중인 거래 정보 저장

    def get_balance(self, currency):
        balances = self.upbit.get_balances()
        for balance in balances:
            if balance['currency'] == currency:
                if balance['balance'] is not None:
                    return float(balance['balance'])
        return 0.0

    def execute_trade(self, signal, current_price):
        if signal == 1 and self.current_trade is None:
            # 매수: KRW 잔고의 30%로 BTC 매수
            krw_balance = self.get_balance("KRW")
            investment = krw_balance * 0.3
            try:
                buy_result = self.upbit.buy_market_order("KRW-BTC", investment)
                if buy_result and buy_result.get('executed_volume'):
                    bought_amount = float(buy_result['executed_volume'])
                    self.current_trade = {
                        'Trade ID': str(uuid.uuid4()),
                        'Ticker': "KRW-BTC",
                        'Entry Time': datetime.now(),
                        'Signal Type': 'Buy',
                        'Entry Price': current_price,
                        'Volume': bought_amount,
                        'Log Close Price': self.trading_system.mrha_data['mh_close'].iloc[-1],
                        'Resistance Level': self.trading_system.mrha_data['Btrg'].iloc[-1],
                        'Support Level': self.trading_system.mrha_data['Stop_Loss'].iloc[-1],
                        'ATR': self.trading_system.mrha_data['ATR'].iloc[-1],
                        'Reason for Entry': "Bullish candle and above Btrg"
                    }
                    send_telegram_message(
                        f"매수 완료: {bought_amount:.6f} BTC @ {current_price:.2f} KRW"
                    )
                    logger.info(f"매수 주문 완료: {bought_amount} BTC @ {current_price} KRW")
            except Exception as e:
                logger.error(f"매수 주문 실패: {e}")

        elif signal == -1 and self.current_trade is not None:
            # 매도: 현재 거래에서 매수한 BTC 전부 매도
            try:
                sell_result = self.upbit.sell_market_order("KRW-BTC", self.current_trade['Volume'])
                if sell_result and sell_result.get('executed_volume'):
                    sold_amount = float(sell_result['executed_volume'])
                    exit_price = current_price
                    exit_time = datetime.now()
                    profit_loss = (exit_price - self.current_trade['Entry Price']) * sold_amount
                    return_pct = (profit_loss / (self.current_trade['Entry Price'] * sold_amount)) * 100

                    trade_record = {
                        'Trade ID': self.current_trade['Trade ID'],
                        'Ticker': self.current_trade['Ticker'],
                        'Entry Time': self.current_trade['Entry Time'],
                        'Exit Time': exit_time,
                        'Signal Type': 'Sell',
                        'Entry Price': self.current_trade['Entry Price'],
                        'Exit Price': exit_price,
                        'Log Close Price': self.trading_system.mrha_data['mh_close'].iloc[-1],
                        'Resistance Level': self.current_trade['Resistance Level'],
                        'Support Level': self.current_trade['Support Level'],
                        'Volume': sold_amount,
                        'ATR': self.current_trade['ATR'],
                        'Reason for Exit': "Stop Loss triggered" if exit_price < self.current_trade['Support Level'] else "Take Profit reached",
                        'Profit/Loss': profit_loss,
                        'Return (%)': return_pct
                    }

                    log_to_notion(trade_record)

                    send_telegram_message(
                        f"매도 완료: {sold_amount:.6f} BTC @ {exit_price:.2f} KRW | MTM: {profit_loss:.2f} KRW"
                    )
                    logger.info(f"매도 주문 완료: {sold_amount} BTC @ {exit_price} KRW | MTM: {profit_loss} KRW")

                    # 현재 거래 초기화
                    self.current_trade = None
            except Exception as e:
                logger.error(f"매도 주문 실패: {e}")

    def run(self):
        logger.info("트레이딩 봇 시작")
        send_telegram_message("MRHA Trading Strategy 시작되었습니다.")

        while True:
            try:
                now = datetime.utcnow()
                # 한국 시간으로 변환 (UTC+9)
                now_kst = now + timedelta(hours=9)
                # 매일 오전 9시에 실행
                if now_kst.hour == 9 and now_kst.minute == 0:
                    logger.info("새로운 캔들 업데이트 감지. 트레이딩 로직 실행 중...")
                    send_telegram_message("새로운 캔들이 생성되었습니다. MRHA Trading Strategy를 실행합니다.")

                    # 데이터 다운로드 및 분석
                    self.trading_system.run_analysis()

                    # 최근 신호 확인
                    latest_signal = self.trading_system.mrha_data['Signal'].iloc[-1]
                    current_price = pyupbit.get_current_price("KRW-BTC")

                    # MTM 계산
                    if self.current_trade:
                        mtm = (current_price - self.current_trade['Entry Price']) * self.current_trade['Volume']
                    else:
                        mtm = 0.0

                    # Telegram 메시지 작성
                    signal_status = "매수 신호 있음" if latest_signal == 1 else ("매도 신호 있음" if latest_signal == -1 else "신호 없음")
                    telegram_message = (
                        f"MRHA Trading Strategy 실행됨\n"
                        f"현재 시각: {now_kst.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"신호 상태: {signal_status}\n"
                        f"종가: {self.trading_system.mrha_data['mh_close'].iloc[-1]:.2f} KRW\n"
                        f"현재 가격: {current_price:.2f} KRW\n"
                        f"보유 BTC 수량: {self.current_trade['Volume']:.6f} BTC" if self.current_trade else "보유 BTC 수량: 0.000000 BTC\n"
                        f"MTM: {mtm:.2f} KRW"
                    )
                    send_telegram_message(telegram_message)
                    logger.info(f"현재 신호: {latest_signal}, 현재 가격: {current_price}, MTM: {mtm}")

                    # 거래 실행
                    self.execute_trade(latest_signal, current_price)

                    # 다음 날까지 대기 (1분 대기하여 중복 실행 방지)
                    time.sleep(60)
                else:
                    # 9시에 실행되기 전에는 대기
                    time.sleep(30)
            except Exception as e:
                logger.error(f"예기치 않은 오류 발생: {e}")
                send_telegram_message(f"오류 발생: {e}")
                time.sleep(60)  # 오류 발생 시 잠시 대기 후 재시도

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
