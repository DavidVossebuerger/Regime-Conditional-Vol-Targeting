from volatility_backtester import VolatilityBacktester


def test_load_all_pattern():
    vb = VolatilityBacktester(data_dir='Kursdaten')
    df = vb.load_data(pattern='*')
    assert not df.empty


def test_load_substring_pattern():
    vb = VolatilityBacktester(data_dir='Kursdaten')
    df = vb.load_data(pattern='xau')
    assert not df.empty


def test_load_literal_pattern():
    vb = VolatilityBacktester(data_dir='Kursdaten')
    # use a filename substring from the docs
    df = vb.load_data(pattern='tslaususd')
    assert not df.empty
