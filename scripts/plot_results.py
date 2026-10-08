"""Plot equity curves from one or more freshly generated walk-forward CSVs."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Compare generated research equity curves")
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1]/"artifacts")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = sorted(args.artifacts.glob("*_walk_forward_backtest.csv"))
    if not paths:
        raise SystemExit(f"No *_walk_forward_backtest.csv in {args.artifacts}; run the model first")
    fig, ax = plt.subplots(figsize=(11,5))
    for path in paths:
        df = pd.read_csv(path, parse_dates=["datetime"])
        if not {"datetime","equity"}.issubset(df.columns):
            continue
        ax.plot(df.datetime, df.equity, label=path.name.replace('_walk_forward_backtest.csv',''))
    ax.set(title="Walk-forward research equity (idealized execution)", ylabel="Net asset value")
    ax.legend();ax.grid(alpha=.15);fig.autofmt_xdate();fig.tight_layout()
    output=args.output or args.artifacts/"walk_forward_equity_comparison.png"
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=165)
    plt.close(fig)
    print(output)

if __name__=="__main__":main()
