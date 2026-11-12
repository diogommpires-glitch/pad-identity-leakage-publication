"""Check the standalone asymmetric overlap CLI with synthetic records."""

import json
import subprocess
import sys
from pathlib import Path


def test_overlap_uses_unique_test_identities(tmp_path):
    train = tmp_path / "train.json"
    test = tmp_path / "test.json"
    train.write_text(json.dumps(["alpha_f00001.png", "alpha_f00002.png"]))
    test.write_text(json.dumps(["alpha_f00003.png", "beta_f00001.png"]))
    script = Path(__file__).resolve().parents[1] / "tool" / "compute_L.py"
    result = subprocess.run(
        [sys.executable, str(script), str(train), str(test), "--key", "video"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "train_ids=1  test_ids=2  overlap=1  L = 0.500" in result.stdout
