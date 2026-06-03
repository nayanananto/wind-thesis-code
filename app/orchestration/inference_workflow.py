import pandas as pd

from app.models.baseline.lstm_baseline_service import LSTMBaselineService
from app.models.power.regression_service import RegressionPowerService


def run_baseline_wind_forecast(
    history_df: pd.DataFrame,
    hours: int,
    bundle_path: str = "models/lstm_model.pkl",
    exog_mode: str = "persistence",
) -> pd.DataFrame:
    service = LSTMBaselineService(bundle_path=bundle_path, exog_mode=exog_mode)
    return service.predict(history_df, hours=hours)


def run_power_mapping(forecast_df: pd.DataFrame, model: object) -> pd.DataFrame:
    service = RegressionPowerService(model=model)
    return service.predict(forecast_df)
