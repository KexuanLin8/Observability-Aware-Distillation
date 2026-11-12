"""The Hydra entry points must register the ppo_haic_* algo configs before composing."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('module', ['research.preflight', 'research.rollout'])
def test_entrypoint_registers_algo_configs(module):
    code = (f'import {module}; from hydra.core.config_store import ConfigStore; '
            "algo = ConfigStore.instance().repo['algo']; "
            "assert 'ppo_haic_train.yaml' in algo and 'ppo_haic_finetune.yaml' in algo, sorted(algo)")
    p = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
