"""Offline regressions for checkpoint and notification input boundaries."""

import ast
import importlib
import pickle
import sys
import types
import zipfile
from http.client import BadStatusLine, LineTooLong
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SENDERS = (
    "effort_seed_robustness",
    "effort_seed_extend",
    "effort_cdf_seed_check",
    "effort_scratch_headroom",
    "effort_ffpp_second_ood",
)


@pytest.fixture
def load_model(monkeypatch, tmp_path):
    source = ROOT / "experiment" / "_dodo_effort_score.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_model"
    )
    module = types.ModuleType("detectors.effort_detector")

    class Detector(torch.nn.Linear):
        def __init__(self, cfg):
            super().__init__(2, 1)

    module.EffortDetector = Detector
    monkeypatch.setitem(sys.modules, "detectors", types.ModuleType("detectors"))
    monkeypatch.setitem(sys.modules, "detectors.effort_detector", module)
    namespace = {"torch": torch, "yaml": yaml, "sys": sys}
    exec(  # noqa: S102 - isolate the trusted local loader from unavailable detector imports.
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    config = tmp_path / "effort.yaml"
    config.write_text("{}\n", encoding="utf-8")
    return lambda weights: namespace["_load_model"](weights, config, "cpu")


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_model_accepts_tensor_state_dicts(load_model, tmp_path, wrapped):
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.bias.copy_(torch.tensor([3.0]))
    checkpoint = model.state_dict()
    if wrapped:
        checkpoint = {
            "state_dict": {"module." + key: value for key, value in checkpoint.items()}
        }
    weights = tmp_path / "weights.pth"
    torch.save(checkpoint, weights)

    loaded = load_model(weights)

    torch.testing.assert_close(
        loaded(torch.tensor([[2.0, 3.0]])), torch.tensor([[11.0]])
    )
    assert loaded.training is False


def _write_marker(path):
    Path(path).write_text("checkpoint executed", encoding="utf-8")
    return {"weight": torch.zeros((1, 2)), "bias": torch.zeros(1)}


class ExecutableCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return _write_marker, (str(self.marker),)


def test_load_model_rejects_executable_checkpoint_without_running_it(
    load_model, tmp_path
):
    marker = tmp_path / "unpickled.txt"
    weights = tmp_path / "malicious.pth"
    torch.save(ExecutableCheckpoint(marker), weights)

    try:
        with pytest.raises(pickle.UnpicklingError):
            load_model(weights)
    finally:
        assert not marker.exists()


@pytest.fixture
def notification_transport(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiment"))
    monkeypatch.delenv("DODO_NTFY_TOPIC", raising=False)
    monkeypatch.delenv("DODO_NTFY_TOKEN", raising=False)
    requests = []

    def capture(request, timeout):
        requests.append(request)
        return types.SimpleNamespace(close=lambda: None)

    monkeypatch.setattr("urllib.request.urlopen", capture)
    return requests


@pytest.mark.parametrize("sender", SENDERS)
@pytest.mark.parametrize("configured", [None, "topic", "token"])
def test_notifications_need_topic_and_token(
    sender, configured, monkeypatch, notification_transport
):
    if configured == "topic":
        monkeypatch.setenv("DODO_NTFY_TOPIC", "private-research-0123456789abcdef")
    elif configured == "token":
        monkeypatch.setenv("DODO_NTFY_TOKEN", "fixture-token")

    importlib.import_module(sender).ntfy("Run complete", "AUC 0.9")

    assert notification_transport == []


@pytest.mark.parametrize(
    "topic",
    (
        "public",
        "old-channel",
        "a" * 31,
        "../other",
        "two/topics",
        "has space",
        "https://host/topic",
        "x" * 65,
    ),
)
def test_notifications_reject_short_or_invalid_topics(
    topic, monkeypatch, notification_transport
):
    monkeypatch.setenv("DODO_NTFY_TOPIC", topic)
    monkeypatch.setenv("DODO_NTFY_TOKEN", "fixture-token")

    importlib.import_module("effort_scratch_headroom").ntfy("Run complete", "AUC 0.9")

    assert notification_transport == []


@pytest.mark.parametrize("sender", SENDERS)
def test_configured_notifications_preserve_destination_and_auth(
    sender, monkeypatch, notification_transport
):
    monkeypatch.setenv("DODO_NTFY_TOPIC", "private-research-0123456789abcdef")
    monkeypatch.setenv("DODO_NTFY_TOKEN", "fixture-token")

    importlib.import_module(sender).ntfy("Run complete", "AUC 0.9")

    assert len(notification_transport) == 1
    request = notification_transport[0]
    assert request.full_url == "https://ntfy.sh/private-research-0123456789abcdef"
    assert request.get_header("Authorization") == "Bearer fixture-token"
    assert request.get_header("Title") == "Run complete"
    assert request.data == b"AUC 0.9"


def test_dense_wrapper_keeps_the_configured_notifier(
    monkeypatch, notification_transport
):
    monkeypatch.setenv("DODO_NTFY_TOPIC", "private-research-0123456789abcdef")
    monkeypatch.setenv("DODO_NTFY_TOKEN", "fixture-token")

    importlib.import_module("effort_scratch_dense").ntfy(
        "Dense result", "AUC 0.9", tags="tada"
    )

    assert len(notification_transport) == 1
    assert notification_transport[0].get_header("Tags") == "tada"


@pytest.mark.parametrize("topic", ["a" * 32, "a" * 64])
def test_notifications_accept_topic_length_boundaries(
    topic, monkeypatch, notification_transport
):
    monkeypatch.setenv("DODO_NTFY_TOPIC", topic)
    monkeypatch.setenv("DODO_NTFY_TOKEN", "fixture-token")

    importlib.import_module("effort_scratch_headroom").ntfy("Run complete", "AUC 0.9")

    assert len(notification_transport) == 1
    assert notification_transport[0].full_url == "https://ntfy.sh/" + topic


@pytest.mark.parametrize(
    "error", [BadStatusLine("invalid status"), LineTooLong("status line")]
)
def test_notification_http_failures_do_not_abort_experiments(
    error, monkeypatch, notification_transport, capsys
):
    monkeypatch.setenv("DODO_NTFY_TOPIC", "private-research-0123456789abcdef")
    monkeypatch.setenv("DODO_NTFY_TOKEN", "fixture-token")

    def fail(request, timeout):
        raise error

    monkeypatch.setattr("urllib.request.urlopen", fail)

    importlib.import_module("effort_scratch_headroom").ntfy("Run complete", "AUC 0.9")

    assert capsys.readouterr().out == f"[--] ntfy failed: {type(error).__name__}\n"


@pytest.mark.parametrize("from_environment", [False, True])
def test_supplementary_archive_is_explicit_and_preserves_splits(
    from_environment, monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(ROOT / "experiment"))
    archive = tmp_path / "supplementary.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "splits/train_leaky_arm.json", '[{"path": "live/a.jpg", "label": 0}]'
        )
        bundle.writestr(
            "splits/train_clean_arm_L0.json", '[{"path": "spoof/b.jpg", "label": 1}]'
        )
        bundle.writestr(
            "splits/test_fixed_heldout.json", '["C:\\\\dataset\\\\live\\\\c.jpg"]'
        )
    module = importlib.import_module("effort_scratch_headroom")
    if from_environment:
        monkeypatch.setenv("DODO_SUPPLEMENTARY_ZIP", str(archive))
        pools, records = module.load_splits()
    else:
        monkeypatch.delenv("DODO_SUPPLEMENTARY_ZIP", raising=False)
        pools, records = module.load_splits(archive)

    assert pools == {
        "leaky": [{"path": "live/a.jpg", "label": 0}],
        "clean": [{"path": "spoof/b.jpg", "label": 1}],
    }
    assert records == [{"path": "C:\\dataset\\live\\c.jpg", "label": 0}]


def test_supplementary_archive_requires_configuration(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiment"))
    monkeypatch.delenv("DODO_SUPPLEMENTARY_ZIP", raising=False)

    with pytest.raises(ValueError, match="supplementary"):
        importlib.import_module("effort_scratch_headroom").load_splits()
