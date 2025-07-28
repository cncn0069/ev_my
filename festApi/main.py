from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import torch
import pandas as pd

# 데이터 및 모델 초기화 변수 (전역)
tft = None
trainer = None
training_dataset = None
group_cols = ["your", "group_cols"]  # 실제 그룹 컬럼명으로 변경 필요

app = FastAPI()

# API 입력 데이터 모델 (예: 예측용 입력 한 건)
class EVChargeAPIInput(BaseModel):
    # 수치형
    last_charge_end_time_ts: float
    connection_start_time_ts: float
    charging_start_time_ts: float
    charging_end_time_ts: float
    connection_end_time_ts: float
    expected_departure_time_ts: float
    idle_time_ts: float
    expected_usage_duration_ts: float
    expected_time_diff_ts: float
    actual_usage_duration_ts: float
    actual_charging_duration_ts: float
    start_delay_duration_ts: float
    post_charge_departure_delay_ts: float
    usage_departure_time_diff_ts: float
    duration_per_kwh_ts: float
    delivered_kwh: float
    kwh_request_diff: float
    kwh_per_usage_time: float

    # 결측치 플래그
    charging_start_time_missing: bool
    charging_end_time_missing: bool
    expected_departure_time_missing: bool
    expected_usage_duration_missing: bool
    expected_time_diff_missing: bool
    actual_charging_duration_missing: bool
    start_delay_duration_missing: bool
    post_charge_departure_delay_missing: bool
    usage_departure_time_diff_missing: bool
    duration_per_kwh_missing: bool
    kwh_per_usage_time_missing: bool

    # 범주형
    station_location: Optional[str]
    evse_name: Optional[str]
    evse_type: Optional[str]
    supports_discharge: Optional[str]

    # 원핫 or 정수형 범주
    scheduled_charge: int
    weekday: int
    usage_departure_range: int
    post_charge_departure_range: int
    cluster: int
    requested_kwh : int

# 예측 결과 반환용 모델
class PredictionOutput(BaseModel):
    prediction: float  # 또는 여러 개라면 리스트로

@app.on_event("startup")
def load_resources():
    global training_dataset, tft, trainer

    # training_dataset 초기화 (training 데이터 준비 후, TimeSeriesDataSet 생성 필요)
    # 예시: training_dataset = TimeSeriesDataSet(...) # 위 코드 참고
    
    # trainer, 모델 초기화 (초기 학습 안 하고 예시)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=0.1,
        enable_model_summary=True,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        optimizer="ranger",
        reduce_on_plateau_patience=4,
    )

@app.post("/train")
def train():
    global trainer, tft, training_dataset
    try:
        # trainer.fit에 validation dataloader 지정 가능
        # val_dataloader = ...
        trainer.fit(tft, train_dataloaders=training_dataset.to_dataloader(train=True, batch_size=64))
        return {"message": "Training complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=List[PredictionOutput])
def predict(inputs: List[EVChargeAPIInput]):
    global tft, training_dataset
    try:
        # 입력 데이터를 DataFrame 형태로 변환 후, TimeSeriesDataSet 변환 필요
        
        # Convert inputs to DataFrame
        input_df = pd.DataFrame([item.dict() for item in inputs])

        # validation / prediction용 TimeSeriesDataSet 생성
        predict_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, input_df, predict=True, stop_randomization=True
        )

        # DataLoader 생성
        predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=len(input_df))

        # 모델 예측, batch 단위로 (보통 1 batch 예상)
        raw_predictions, x = tft.predict(predict_dataloader, mode="raw", return_x=True)

        # 예: raw_predictions['prediction'] 혹은 'quantiles' 등에서 원하는 값 추출
        # 간단 예시에서는 median 혹은 특정 quantile 값을 prediction으로 반환
        preds = tft.predict(predict_dataloader)

        results = [{"prediction": float(p)} for p in preds]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
