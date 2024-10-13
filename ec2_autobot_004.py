import os
import time
import uuid
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import pyupbit
import pandas as pd
import numpy as np

# 환경 변수 로드
load_dotenv()

# 설정
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')
NOTION_API_TOKEN = os.getenv('NOTION_API_TOKEN')
NOTION_DATABASE_ID = os.getenv('NOTION_DATABASE_ID')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')  # Telegram 관련 변수 제거

# 필수 환경 변수 검증
required_env_vars = ['UPBIT_ACCESS_KEY', 'UPBIT_SECRET_KEY', 'NOTION_API_TOKEN', 'NOTION_DATABASE_ID', 'SLACK_WEBHOOK_URL']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('trading_bot.log')  # 로그 파일 핸들러 추가
                    ])
logger = logging.getLogger()

def send_slack_message(message: str):
    """Slack 메시지 전송 함수"""
    if not SLACK_WEBHOOK_URL:
        logger.error("Slack Webhook URL이 설정되지 않았습니다.")
        return
    try:
        payload = {
            "text": message
        }
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code != 200:
            logger.error(f"Slack 메시지 전송 실패: {response.status_code}, {response.text}")
        else:
            logger.info("Slack 메시지 전송 완료")
    except Exception as e:
        logger.error(f"Slack 메시지 전송 중 예외 발생: {e}")

def log_to_notion(trade: dict):
    """Notion 데이터베이스에 거래 기록"""
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
            "Exit Time": {"date": {"start": trade['Exit Time'].isoformat()}} if trade['Exit Time'] else {},
            "Signal Type": {"select": {"name": trade['Signal Type']}},
            "Entry Price": {"number": trade['Entry Price']},
            "Exit Price": {"number": trade['Exit Price']} if trade['Exit Price'] else {},
            "Volume": {"number": trade['Volume']},
            "Stop Loss": {"number": trade['Stop Loss']},
            "Take Profit": {"number": trade['Take Profit']},
            "Status": {"select": {"name": trade['Status']}},
            "Profit/Loss": {"number": trade.get('Profit/Loss', 0)},
            "Return (%)": {"number": trade.get('Return (%)', 0)},
            "Reason for Entry": {"rich_text": [{"text": {"content": trade.get('Reason for Entry', '')}}]},
            "Reason for Exit": {"rich_text": [{"text": {"content": trade.get('Reason for Exit', '')}}]} if trade.get('Reason for Exit') else {},
            "Notes": {"rich_text": [{"text": {"content": trade.get('Notes', '')}}]} if trade.get('Notes') else {},
        }
        # 비어있는 속성 제거
        properties = {k: v for k, v in properties.items() if v != {}}
        data = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": properties
        }
        logger.debug(f"Properties being sent to Notion: {properties}")
        response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"Notion 데이터베이스 업데이트 실패: {response.text}")
        else:
            logger.info("Notion 데이터베이스 업데이트 완료")
            send_slack_message("Notion 데이터베이스에 거래가 기록되었습니다.")
    except Exception as e:
        logger.error(f"Notion 데이터베이스에 기록 중 예외 발생: {e}")

def update_notion_trade(trade_id: str, exit_price: float, exit_time: datetime, profit_loss: float, return_pct: float, reason_for_exit: str):
    """Notion 데이터베이스의 특정 거래 업데이트"""
    try:
        headers = {
            "Authorization": f"Bearer {NOTION_API_TOKEN}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        # 먼저 페이지 ID를 찾는 쿼리 수행
        query = {
            "filter": {
                "property": "Trade ID",
                "title": {
                    "equals": str(trade_id)
                }
            }
        }
        response = requests.post(f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query", headers=headers, json=query)
        if response.status_code != 200:
            logger.error(f"Notion 데이터베이스 조회 실패: {response.text}")
            return
        results = response.json().get('results', [])
        if not results:
            logger.error("해당 Trade ID를 찾을 수 없습니다.")
            return
        page_id = results[0]['id']
        # 업데이트할 속성 정의
        properties = {
            "Exit Time": {"date": {"start": exit_time.isoformat()}},
            "Exit Price": {"number": exit_price},
            "Profit/Loss": {"number": profit_loss},
            "Return (%)": {"number": return_pct},
            "Signal Type": {"select": {"name": "Sell"}},
            "Reason for Exit": {"rich_text": [{"text": {"content": reason_for_exit}}]},
            "Status": {"select": {"name": "Closed"}}
        }
        update_data = {
            "properties": properties
        }
        update_response = requests.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=headers, json=update_data)
        if update_response.status_code != 200:
            logger.error(f"Notion 데이터베이스 업데이트 실패: {update_response.text}")
        else:
            logger.info("Notion 데이터베이스 거래 업데이트 완료")
            send_slack_message(f"Notion 데이터베이스에서 Trade ID: {trade_id}의 거래가 업데이트되었습니다.")
    except Exception as e:
        logger.error(f"Notion 데이터베이스 업데이트 중 예외 발생: {e}")

def get_current_trade_from_notion():
    """Notion 데이터베이스에서 현재 진행 중인 거래 조회"""
    try:
        headers = {
            "Authorization": f"Bearer {NOTION_API_TOKEN}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        query = {
            "filter": {
                "property": "Status",
                "select": {
                    "equals": "Open"
                }
            },
            "sorts": [
                {
                    "property": "Entry Time",
                    "direction": "descending"
                }
            ],
            "page_size": 1
        }
        response = requests.post(f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query", headers=headers, json=query)
        if response.status_code != 200:
            logger.error(f"Notion 데이터베이스 조회 실패: {response.text}")
            return None
        results = response.json().get('results', [])
        if not results:
            return None
        page = results[0]
        properties = page['properties']
        
        # 필드 검증 및 기본값 설정
        def get_property(prop, prop_type, default=None):
            try:
                if prop_type == 'title':
                    return properties[prop]['title'][0]['text']['content'] if properties[prop]['title'] else default
                elif prop_type == 'rich_text':
                    return properties[prop]['rich_text'][0]['text']['content'] if properties[prop]['rich_text'] else default
                elif prop_type == 'date':
                    return datetime.fromisoformat(properties[prop]['date']['start']) if properties[prop]['date'] else default
                elif prop_type == 'select':
                    return properties[prop]['select']['name'] if properties[prop]['select'] else default
                elif prop_type == 'number':
                    return properties[prop]['number'] if properties[prop]['number'] is not None else default
                else:
                    return default
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"필드 {prop} 누락 또는 형식 오류: {e}")
                return default
        
        current_trade = {
            'Trade ID': get_property('Trade ID', 'title', ''),
            'Ticker': get_property('Ticker', 'rich_text', ''),
            'Entry Time': get_property('Entry Time', 'date', None),
            'Signal Type': get_property('Signal Type', 'select', ''),
            'Entry Price': get_property('Entry Price', 'number', 0.0),
            'Volume': get_property('Volume', 'number', 0.0),
            'Stop Loss': get_property('Stop Loss', 'number', 0.0),
            'Take Profit': get_property('Take Profit', 'number', 0.0),
            'Status': get_property('Status', 'select', '')
        }
        return current_trade
    except Exception as e:
        logger.error(f"Notion 데이터베이스에서 현재 거래 조회 중 예외 발생: {e}")
        return None

def calculate_atr(high, low, close, period):
    """ATR (Average True Range) 계산 함수"""
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

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
        
        # h_close 계산
        ha['h_close'] = (ha['h_open'] + ha['h_high'] + ha['h_low'] + ha['h_close']) / 4
        
        # h_open 계산 - 벡터화된 방식으로 변경
        ha['h_open'] = (ha['h_open'].shift(1) + ha['h_close'].shift(1)) / 2
        ha['h_open'].iloc[0] = ha['h_open'].iloc[0]  # 첫 번째 h_open 값 유지
        
        # h_high, h_low 계산
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
        # Fibonacci 수준 기반 신호 계산
        signals = pd.DataFrame(index=self.mrha_data.index)
        signals['Ebr'] = (4 * self.mrha_data['mh_open'] - self.stock_data['Low']) / 3
        signals['Btrg'] = 1.00618 * signals['Ebr']
        signals['Ebl'] = (4 * self.mrha_data['mh_open'] - self.stock_data['High']) / 3
        signals['Strg'] = 0.99382 * signals['Ebl']

        # ATR 계산
        signals['ATR'] = calculate_atr(self.stock_data['High'], self.stock_data['Low'], self.stock_data['Close'], self.atr_period)

        # Stop Loss 및 Take Profit 레벨 설정
        signals['Stop_Loss'] = self.mrha_data['mh_close'] - (signals['ATR'] * 1.5)
        signals['Take_Profit'] = self.mrha_data['mh_close'] + (signals['ATR'] * 3)

        self.mrha_data = pd.concat([self.mrha_data, signals], axis=1)

    def implement_trading_logic(self):
        # 벡터화된 거래 신호 생성
        self.mrha_data['Signal'] = 0
        # Bullish candle 조건
        bullish_candle = (
            (self.mrha_data['mh_close'] > self.mrha_data['mh_open']) &
            (self.mrha_data['mh_close'] > self.mrha_data['Btrg'].shift(1))
        )
        # 매수 신호
        self.mrha_data.loc[bullish_candle, 'Signal'] = 1

        # 매도 신호 조건
        condition_sell = (
            (self.mrha_data['mh_close'] < self.mrha_data['Stop_Loss']) |
            (self.mrha_data['mh_close'] > self.mrha_data['Take_Profit'])
        )
        # 매도 신호
        self.mrha_data.loc[condition_sell, 'Signal'] = -1

        # 포지션 관리 로직 추가
        # 포지션이 열려 있을 때만 매도 신호를 유지
        self.mrha_data['Position'] = 0
        self.mrha_data['Position'] = self.mrha_data['Signal'].replace(0, np.nan).ffill().fillna(0)
        self.mrha_data.loc[self.mrha_data['Position'] > 0, 'Signal'] = self.mrha_data['Signal']

        # 이제 Signal은 매수 시 1, 매도 시 -1, 그 외에는 0

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
        self.last_trade_date = None  # 마지막 거래 실행 날짜 저장

    def get_balance(self, currency):
        balances = self.upbit.get_balances()
        for balance in balances:
            if balance['currency'] == currency:
                if balance['balance'] is not None:
                    return float(balance['balance'])
        return 0.0

    def execute_trade(self, signal, current_price):
        if signal == 1 and not self.current_trade:
            # 매수: KRW 잔고의 30%로 BTC 매수
            krw_balance = self.get_balance("KRW")
            investment = krw_balance * 0.3
            try:
                buy_result = self.upbit.buy_market_order("KRW-BTC", investment)
                if buy_result and buy_result.get('executed_volume'):
                    bought_amount = float(buy_result['executed_volume'])
                    # 필드 검증
                    stop_loss = self.trading_system.mrha_data['Stop_Loss'].iloc[-2] if 'Stop_Loss' in self.trading_system.mrha_data.columns else 0.0
                    take_profit = self.trading_system.mrha_data['Take_Profit'].iloc[-2] if 'Take_Profit' in self.trading_system.mrha_data.columns else 0.0
                    self.current_trade = {
                        'Trade ID': str(uuid.uuid4()),
                        'Ticker': "KRW-BTC",
                        'Entry Time': datetime.utcnow(),
                        'Signal Type': 'Buy',
                        'Entry Price': current_price,
                        'Volume': bought_amount,
                        'Stop Loss': stop_loss,
                        'Take Profit': take_profit,
                        'Status': 'Open'
                    }
                    # Stop Loss 및 Take Profit 검증
                    if not self.current_trade['Volume'] or not self.current_trade['Stop Loss'] or not self.current_trade['Take Profit']:
                        logger.error("매수 후 거래 정보 누락: Volume, Stop Loss 또는 Take Profit이 설정되지 않았습니다.")
                        send_slack_message("매수 후 거래 정보 누락: Volume, Stop Loss 또는 Take Profit이 설정되지 않았습니다.")
                        self.current_trade = None
                        return
                    log_to_notion(self.current_trade)
                    send_slack_message(
                        f"매수 완료: {bought_amount:.6f} BTC @ {current_price:.2f} KRW"
                    )
                    logger.info(f"매수 주문 완료: {bought_amount} BTC @ {current_price:.2f} KRW")
            except Exception as e:
                logger.error(f"매수 주문 실패: {e}")

        elif signal == -1 and self.current_trade:
            # 매도: 현재 거래에서 매수한 BTC 전부 매도
            try:
                sell_result = self.upbit.sell_market_order("KRW-BTC", self.current_trade['Volume'])
                if sell_result and sell_result.get('executed_volume'):
                    sold_amount = float(sell_result['executed_volume'])
                    exit_price = current_price
                    exit_time = datetime.utcnow()
                    profit_loss = (exit_price - self.current_trade['Entry Price']) * sold_amount
                    return_pct = (profit_loss / (self.current_trade['Entry Price'] * sold_amount)) * 100

                    # 매도 이유 결정
                    if exit_price < self.current_trade.get('Stop Loss', 0.0):
                        reason_for_exit = "Stop Loss triggered"
                    elif exit_price >= self.current_trade.get('Take Profit', 0.0):
                        reason_for_exit = "Take Profit reached"
                    else:
                        reason_for_exit = "Sell signal triggered"

                    # 거래 기록 업데이트
                    update_notion_trade(
                        trade_id=self.current_trade['Trade ID'],
                        exit_price=exit_price,
                        exit_time=exit_time,
                        profit_loss=profit_loss,
                        return_pct=return_pct,
                        reason_for_exit=reason_for_exit
                    )

                    send_slack_message(
                        f"매도 완료: {sold_amount:.6f} BTC @ {exit_price:.2f} KRW | MTM: {profit_loss:.2f} KRW"
                    )
                    logger.info(f"매도 주문 완료: {sold_amount} BTC @ {exit_price:.2f} KRW | MTM: {profit_loss:.2f} KRW")

                    # 현재 거래 초기화
                    self.current_trade = None
            except Exception as e:
                logger.error(f"매도 주문 실패: {e}")

    def run(self):
        logger.info("트레이딩 봇 시작")
        send_slack_message("MRHA Trading Strategy 시작되었습니다.")

        while True:
            try:
                now = datetime.utcnow()
                # 한국 시간으로 변환 (UTC+9)
                now_kst = now + timedelta(hours=9)
                today_date = now_kst.date()

                # 매일 오전 9시에 실행
                if now_kst.hour == 9 and now_kst.minute == 0:
                    # 거래가 이미 실행된 날인지 확인
                    if self.last_trade_date != today_date:
                        logger.info("새로운 캔들 업데이트 감지. 트레이딩 로직 실행 중...")
                        send_slack_message("새로운 캔들이 생성되었습니다. MRHA Trading Strategy를 실행합니다.")

                        # 데이터 다운로드 및 분석
                        self.trading_system.run_analysis()

                        # 이전 캔들의 신호 확인 (마지막 캔들이 새로운 캔들이기 때문에 이전 캔들을 참조)
                        if 'Signal' not in self.trading_system.mrha_data.columns or len(self.trading_system.mrha_data['Signal']) < 2:
                            logger.warning("신호 데이터가 부족하여 트레이딩을 실행하지 않습니다.")
                            send_slack_message("신호 데이터가 부족하여 트레이딩을 실행하지 않습니다.")
                            time.sleep(60)
                            continue

                        previous_signal = self.trading_system.mrha_data['Signal'].iloc[-2]
                        current_price = pyupbit.get_current_price("KRW-BTC")

                        # Stop Loss 및 Take Profit 레벨 가져오기
                        stop_loss = self.trading_system.mrha_data['Stop_Loss'].iloc[-2] if 'Stop_Loss' in self.trading_system.mrha_data.columns else None
                        take_profit = self.trading_system.mrha_data['Take_Profit'].iloc[-2] if 'Take_Profit' in self.trading_system.mrha_data.columns else None

                        # Stop Loss 및 Take Profit 검증
                        if stop_loss is None or take_profit is None:
                            logger.error("Stop Loss 또는 Take Profit 정보가 누락되었습니다. 매도 신호를 무시합니다.")
                            send_slack_message("Stop Loss 또는 Take Profit 정보가 누락되었습니다. 매도 신호를 무시합니다.")
                            time.sleep(60)
                            continue

                        # 현재 포지션 조회 (Notion 데이터베이스에서)
                        current_trade = get_current_trade_from_notion()
                        self.current_trade = current_trade

                        # MTM 계산
                        if self.current_trade:
                            mtm = (current_price - self.current_trade['Entry Price']) * self.current_trade['Volume']
                        else:
                            mtm = 0.0

                        # Slack 메시지 작성
                        if self.current_trade:
                            slack_message = (
                                f"MRHA Trading Strategy 실행됨\n"
                                f"현재 시각: {now_kst.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"신호 상태: {'매수 신호 있음' if previous_signal == 1 else ('매도 신호 있음' if previous_signal == -1 else '신호 없음')}\n"
                                f"종가: {self.trading_system.mrha_data['mh_close'].iloc[-2]:.2f} KRW\n"
                                f"현재 가격: {current_price:.2f} KRW\n"
                                f"보유 BTC 수량: {self.current_trade['Volume']:.6f} BTC\n"
                                f"MTM: {mtm:.2f} KRW"
                            )
                        else:
                            slack_message = (
                                f"MRHA Trading Strategy 실행됨\n"
                                f"현재 시각: {now_kst.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"신호 상태: {'매수 신호 있음' if previous_signal == 1 else ('매도 신호 있음' if previous_signal == -1 else '신호 없음')}\n"
                                f"종가: {self.trading_system.mrha_data['mh_close'].iloc[-2]:.2f} KRW\n"
                                f"현재 가격: {current_price:.2f} KRW\n"
                                f"보유 BTC 수량: 0.000000 BTC\n"
                                f"MTM: {mtm:.2f} KRW"
                            )
                        send_slack_message(slack_message)
                        logger.info(f"이전 신호: {previous_signal}, 현재 가격: {current_price}, MTM: {mtm}")

                        # 매수/매도 조건 평가 및 실행
                        # 이전 캔들의 신호에 따라 매수
                        if previous_signal == 1 and not self.current_trade:
                            self.execute_trade(1, current_price)
                        # 현재 포지션이 있는 경우, 매도 조건 평가
                        elif self.current_trade:
                            sell_signal = previous_signal == -1
                            stop_loss_triggered = current_price < stop_loss
                            take_profit_triggered = current_price >= take_profit
                            if sell_signal or stop_loss_triggered or take_profit_triggered:
                                self.execute_trade(-1, current_price)

                        # 마지막 거래 실행 날짜 업데이트
                        self.last_trade_date = today_date

                        # 다음 날까지 대기 (1분 대기하여 중복 실행 방지)
                        time.sleep(60)
                    else:
                        logger.info(f"오늘({today_date}) 이미 트레이딩이 실행되었습니다. 대기 중...")
                        time.sleep(60)  # 이미 실행된 경우 1분 대기
            except Exception as e:
                logger.error(f"예기치 않은 오류 발생: {e}")
                send_slack_message(f"오류 발생: {e}")
                time.sleep(60)  # 오류 발생 시 잠시 대기 후 재시도

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
