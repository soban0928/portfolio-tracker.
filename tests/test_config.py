import pytest

from simulator.config import FeeConfig


def test_fee_config_defaults():
    fees = FeeConfig()
    assert fees.commission == 5.0
    assert fees.slippage_rate == pytest.approx(0.001)


@pytest.mark.parametrize("kwargs", [{"commission": -1}, {"slippage_bps": -0.5}])
def test_fee_config_rejects_negative(kwargs):
    with pytest.raises(ValueError):
        FeeConfig(**kwargs)
