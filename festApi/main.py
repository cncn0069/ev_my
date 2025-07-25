# main.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from sklearn.preprocessing import RobustScaler  # 또는 StandardScaler 등
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    GroupNormalizer,
    MAE,
    QuantileLoss
)

# ----------------------------------------
# ✅ LSTM 모델 클래스 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, dropout=0.1):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------
# 📦 입력 데이터 스키마 정의
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


# ----------------------------------------
# ✅ 설정값 (수정 필요)
INPUT_SIZE = 12
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1
SEQ_LENGTH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------
# ✅ 모델 및 지원 파일 로드
lstm_model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
lstm_model.load_state_dict(torch.load("../model/lstm_model_best.pth", map_location=DEVICE))
lstm_model.eval()

temporal_model =TemporalFusionTransformer.load_from_checkpoint("../model/epoch=0-step=242.ckpt")
temporal_model.eval()



# ✅ 훈련 시 저장한 정보 불러오기
scaler = joblib.load("../model/input_scaler.pkl")          # 입력 feature용
target_scaler = joblib.load("../model/target_scaler.pkl")   # 타깃(requested_kwh) 용
numerical_columns = joblib.load("../model/numeric_cols.pkl")           # 입력 수치형
expected_features = joblib.load("../model/expected_features.pkl")

# 학습 시 입력 순서 유지용
# ✅ MLP 모델 로드
mlp_model = MLPModel(input_size=len(expected_features)).to(DEVICE)
mlp_model.load_state_dict(torch.load("../model/mlp_model_best.pth", map_location=DEVICE))
mlp_model.eval()


categorical_columns = [
    'evse_type','weekday'
]
one_hot_columns = joblib.load("../model/one_hot_columns.pkl")  

# ----------------------------------------
# ✅ FastAPI 앱 실행
app = FastAPI()

@app.get("/")
def index():
    return {"message": "EV Charging Prediction API 💡"}

@app.post("/predict")
def predict(input_data: EVChargeAPIInput):
    try:
        # 입력 딕셔너리를 DataFrame으로
        df = pd.DataFrame([input_data.dict()])
        df.columns = df.columns.str.strip()

        # 수치형 결측치 처리
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 범주형 원핫 인코딩
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

        # 누락된 expected_features 보완 (0으로 채움)
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # 순서 맞춤
        df = df[expected_features]

        # 수치형 스케일링
        inputs_for_scaling = [col for col in numerical_columns if col in df.columns]
        df[inputs_for_scaling] = scaler.transform(df[inputs_for_scaling])

        # LSTM 입력: (1, seq_length, input_size)
        input_np = df.to_numpy().astype(np.float32)
        input_seq = np.repeat(input_np[np.newaxis, :, :], SEQ_LENGTH, axis=1)
        input_tensor = torch.tensor(input_seq).to(DEVICE)

        # 모델 예측 (정규화된 값)
        with torch.no_grad():
            normalized_output = lstm_model(input_tensor).cpu().numpy()[0, 0]

        # ✅ 타깃 스케일러로 역변환
        denormalized_output = target_scaler.inverse_transform([[normalized_output]])[0, 0]

        return {
            "normalized_prediction": float(normalized_output),
            "denormalized_prediction": float(denormalized_output)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))

@app.post("/predict/mlp")
def predict_mlp(input_data: EVChargeAPIInput):
    try:
        # 입력 데이터프레임 생성
        df = pd.DataFrame([input_data.dict()])
        df.columns = df.columns.str.strip()

        # 수치형 결측치 처리
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 범주형 원핫 인코딩
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

        # 누락된 피처 보완 (0으로)
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # 컬럼 순서 정렬
        df = df[expected_features]

        # 스케일링 (수치형만)
        inputs_for_scaling = [col for col in numerical_columns]
        df[inputs_for_scaling] = scaler.transform(df[inputs_for_scaling])
        df = df.astype(np.float32)
        print("🌐 DataFrame dtypes:")
        print(df.dtypes)

        # 🎯 MLP 입력 : (1, input_size)
        input_tensor = torch.tensor(df.values, dtype=torch.float32).to(DEVICE)

        # 예측
        with torch.no_grad():
            normalized_output = mlp_model(input_tensor).item()

        denormalized_output = target_scaler.inverse_transform([[normalized_output]])[0][0]

        return {
            "normalized_prediction": float(normalized_output),
            "denormalized_prediction": float(denormalized_output),
            "model": "MLP"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
    
@app.post("/predict/temporl")
def predict(request: EVChargeAPIInput):
    # 1. 입력 데이터 전처리(DataFrame, Tensor 등)
    input_dict = request.dict()
    input_df = pd.DataFrame([input_dict])
    input_data = input_df  # 입력 데이터 포맷에 맞게 변환
    # 2. 모델 추론
    with torch.no_grad():
        prediction = temporal_model.predict(input_data)
    # 3. 후처리 및 결과 리턴
    return {"result": prediction.tolist()}