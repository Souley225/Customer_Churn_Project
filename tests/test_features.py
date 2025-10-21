
from __future__ import annotations
import pandas as pd
from src.features.build_features import TelcoCleaner

def test_cleaner_creates_features() -> None:
    df = pd.DataFrame({
        "customerID": ["0001"],
        "tenure": [5],
        "MonthlyCharges": [70.0],
        "TotalCharges": [" "],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["No"],
    })
    cl = TelcoCleaner()
    out = cl.fit_transform(df)
    assert "tenure_bucket" in out.columns
    assert "num_services" in out.columns
    assert "total_spend_proxy" in out.columns
    assert out["TotalCharges"].dtype.kind in ("f", "i")
