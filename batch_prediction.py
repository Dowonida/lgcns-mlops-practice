import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.common.constants import (
    ARTIFACT_PATH,
    DATA_PATH,
    LOG_FILEPATH,
    PREDICTION_PATH,
)
from src.common.logger import handle_exception, set_logger

logger = set_logger(os.path.join(LOG_FILEPATH, "logs.log"))
sys.excepthook = handle_exception
warnings.filterwarnings(action="ignore")

if __name__ == "__main__":
    DATE = datetime.now().strftime("%Y%m%d")
    test = pd.read_csv(os.path.join(DATA_PATH, "house_rent_test.csv"))
    model = joblib.load(os.path.join(ARTIFACT_PATH, "model.pkl"))

    X = test.drop(["id", "rent"], axis=1, inplace=False)
    id_ = test["id"].to_numpy()
    y = test["rent"].to_numpy()

    # 테스트 데이터에 대한 피처 데이터 저장
    model["preprocessor"].transform(X=X).to_csv(
        os.path.join(DATA_PATH, "storage", "house_rent_test_feature.csv"),
        index=False,
    )

    pred_df = pd.DataFrame({"user": id_, "rent": np.expm1(model.predict(X))})

    logger.info(f"Batch prediction for {len(pred_df)} users is created.")

    save_path = os.path.join(PREDICTION_PATH, f"{DATE}_rent_prediction.csv")
    pred_df.to_csv(save_path, index=False)
    logger.info(
        "Prediction can be found in the following path:\n" f"{save_path}"
    )
