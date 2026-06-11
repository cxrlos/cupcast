import pandas as pd

from cupcast.report.build import latex_table


def test_latex_table_escapes_underscores_in_values_not_headers():
    frame = pd.DataFrame(
        {"forecaster": ["dixon_coles", "pinnacle_closing"], "log_loss": [0.95, 0.93]}
    )
    rendered = latex_table(
        frame, {"forecaster": "Forecaster ($\\Delta$)", "log_loss": "Log-loss"}
    )
    assert r"dixon\_coles" in rendered
    assert r"pinnacle\_closing" in rendered
    assert "Forecaster ($\\Delta$)" in rendered  # header math left intact
    assert "0.950" in rendered
