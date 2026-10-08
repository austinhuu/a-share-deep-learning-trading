"""Optional VeighNa CSV signal monitor (NO ORDER SUBMISSION).

Requires vnpy + vnpy_ctastrategy in a separate VeighNa environment.
The Stage A signals are offline historical research predictions, not live
intraday signals. To match, bar timestamps must have identical frequency.
"""
from pathlib import Path
import pandas as pd
from vnpy_ctastrategy import CtaTemplate
from vnpy.trader.object import BarData, OrderData, TradeData

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class AIModelStrategy(CtaTemplate):
    author = "Research demo: no live orders"
    model_name = "lstm"
    artifact_dir = str(PROJECT_ROOT / "artifacts")
    buy_threshold = 0.55
    sell_threshold = 0.45
    parameters = ["model_name", "artifact_dir", "buy_threshold", "sell_threshold"]
    pred_prob = 0.5
    last_signal = "none"
    variables = ["pred_prob", "last_signal"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.signal_map = {}

    def on_init(self):
        path = Path(self.artifact_dir) / f"{self.model_name}_signals.csv"
        frame = pd.read_csv(path)
        if not {"datetime", "prob"}.issubset(frame.columns):
            raise ValueError("Signal CSV requires datetime and prob")
        dates = pd.to_datetime(frame["datetime"])
        self.signal_map = dict(zip(dates.dt.strftime("%Y-%m-%d %H:%M:%S"), frame["prob"].astype(float)))
        self.write_log(f"Offline signal monitor loaded {len(self.signal_map)} historical entries")

    def on_start(self):
        self.write_log("Research monitor started (orders disabled)")

    def on_stop(self):
        self.write_log("Research monitor stopped")

    def on_bar(self, bar: BarData):
        key = bar.datetime.strftime("%Y-%m-%d %H:%M:%S")
        p = self.signal_map.get(key)
        if p is None:
            self.last_signal = "no_matching_offline_signal"
            return
        self.pred_prob = float(p)
        self.last_signal = ("buy_intent" if p >= self.buy_threshold
                            else "sell_intent" if p <= self.sell_threshold
                            else "hold_intent")
        self.write_log(f"{key}  p={p:.4f}  {self.last_signal} (SIMULATION ONLY)")
        self.put_event()

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        pass
