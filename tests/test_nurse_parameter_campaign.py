import importlib.util
from pathlib import Path


def _load_campaign_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_nurse_parameter_campaign.py"
    spec = importlib.util.spec_from_file_location("run_nurse_parameter_campaign", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parameter_campaign_defaults_match_current_ablation_protocol():
    module = _load_campaign_module()

    assert module.DEFAULT_ABLATION_MODES == ["full", "wo_E", "wo_P", "wo_H"]
    assert module.DEFAULT_ABLATION_MODELS == [
        "ep-20260615114505-247zc",
        "ep-20260623103535-v7z57",
        "ep-20260623103508-6dkw6",
        "ep-m-20260430194631-77b5x",
        "ep-m-20260430105117-46zl7",
        "ep-20260630151544-g59w4",
        "ep-20260630151630-q6q2k",
    ]
