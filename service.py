import bentoml
import numpy as np
import numpy.typing as npt
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


class Features(BaseModel):
    bhk: int
    size: int
    floor: str
    area_type: str
    city: str
    furnishing_status: str
    tenant_preferred: str
    bathroom: int
    point_of_contact: str


bento_model = bentoml.sklearn.get("house_rent:latest")
model_runner = bento_model.to_runner()

svc = bentoml.Service("rent_house_regressor", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=Features), output=NumpyNdarray())
async def predict(input_data: Features) -> npt.NDArray:
    input_df = pd.DataFrame([input_data.dict()])
    log_pred = await model_runner.predict.async_run(input_df)
    return np.expm1(log_pred)
