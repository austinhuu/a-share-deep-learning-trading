from pathlib import Path
import subprocess
import sys


def test_offline_mlp_walkforward(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    process = subprocess.run([
        sys.executable, str(repo / 'run_stage_a.py'), '--demo', '--demo-bars','220',
        '--mode','walk_forward', '--model','mlp', '--epochs','1', '--patience','1',
        '--batch-size','32','--window','20','--test-bars','80',
        '--device','cpu','--output',str(tmp_path)],cwd=repo,
        text=True,capture_output=True,timeout=180)
    assert process.returncode==0, process.stdout + '\n' + process.stderr
    assert (tmp_path/'mlp_walk_forward_predictions.csv').exists()
    assert (tmp_path/'mlp_walk_forward_metrics.json').exists()
    assert (tmp_path/'mlp_signals.csv').exists()
