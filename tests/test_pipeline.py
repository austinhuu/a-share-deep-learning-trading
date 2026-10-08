import numpy as np
import pandas as pd
import pytest

from backtesting.engine import risk_managed_backtest
from backtesting.metrics import max_drawdown
from backtesting.rolling_walk_forward_backtest import make_fold_arrays
from data.processor import DataProcessor
from data.sample_generator import make_synthetic_ohlcv
from features.feature_engineer import create_features, get_default_feature_cols
from features.label_generator import add_binary_label
from features.preprocess import build_sequences
from training.cv import expanding_window_split


def test_missing_future_labels_are_nan():
    df=pd.DataFrame({'close':[1,2,1,4]})
    result=add_binary_label(df,horizon=1)
    assert result.label.tolist()[:3]==[1.0,0.0,1.0]
    assert np.isnan(result.label.iloc[-1])


def test_sequence_time_alignment():
    df=pd.DataFrame({'datetime':pd.date_range('2024-01-01', periods=5),
                     'close':[1,2,3,4,5], 'feature':np.arange(5,dtype=float),
                     'label':[0.,1.,0.,1.,np.nan]})
    X,y,meta=build_sequences(df,['feature'],window=3)
    assert len(X)==2
    np.testing.assert_array_equal(X[0,:,0],[0,1,2])
    assert meta.iloc[0].row_index==2
    assert pd.Timestamp(meta.iloc[0].datetime)==pd.Timestamp('2024-01-03')


def test_purged_expanding_cv():
    splits=list(expanding_window_split(200,3,purge=1))
    assert len(splits)==3
    for tr,va in splits:
        assert tr[-1]+1 < va[0]  # label for tr[-1] is not in validation


def test_walkforward_scaler_fit_excludes_future_regime():
    df=DataProcessor().standardize_ohlcv(make_synthetic_ohlcv(260))
    df=add_binary_label(create_features(df))
    cols=get_default_feature_cols()
    df=df.dropna(subset=cols+['label']).reset_index(drop=True)
    val_start,train_end,test_end=120,165,195
    arr=make_fold_arrays(df, cols, window=30,val_start=val_start,
                         train_end=train_end,test_end=test_end,purge=1)
    Xt,yt,Xv,yv,Xtest,ytest,meta,scaler=arr
    np.testing.assert_allclose(scaler.mean_,df.iloc[:val_start-1][cols].mean().values,
                               rtol=1e-6,atol=1e-6)
    assert meta.row_index.min()==train_end
    assert len(Xtest)==test_end-train_end
    assert len(Xt)>0 and len(Xv)>0


def test_current_signal_does_not_earn_current_bar_return():
    bt,stats=risk_managed_backtest(pd.Series([100.,110.,121.]), pd.Series([0.9,0.1,0.1]),
                                   commission=0,slippage=0,max_position=1,min_position=1)
    assert bt.strategy_return.iloc[0]==0
    assert bt.strategy_return.iloc[1]==pytest.approx(.1)
    assert bt.strategy_return.iloc[2]==0


def test_costs_charged_only_for_turnover():
    bt, _ = risk_managed_backtest(pd.Series([100.,100.,100.]), pd.Series([.9,.9,.2]),
                                  commission=.001,slippage=0,max_position=1,min_position=1)
    assert bt.trade.tolist()==[1,0,1]
    assert bt.trade_cost.sum()==pytest.approx(.002)


def test_drawdown_includes_initial_equity_one():
    assert max_drawdown([.8,.9,1.0])==pytest.approx(-.2)


def test_tushare_secret_is_not_embedded():
    from pathlib import Path
    loader=(Path(__file__).resolve().parents[1]/'data/loader.py').read_text()
    assert 'os.getenv("TUSHARE_TOKEN")' in loader
    assert 'ts.set_token(token)' in loader


def test_integer_yyyymmdd_is_parsed_correctly():
    raw = pd.DataFrame({"trade_date": [20220105, 20220103], "open": [3., 2.],
                        "high": [4., 3.], "low": [2., 1.], "close": [3., 2.],
                        "vol": [100, 200]})
    out = DataProcessor().standardize_ohlcv(raw)
    assert str(out.datetime.iloc[0].date()) == "2022-01-03"
    assert str(out.datetime.iloc[1].date()) == "2022-01-05"
