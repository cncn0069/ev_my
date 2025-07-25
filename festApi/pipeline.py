from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def model_pipe():
    # 수치형/범주형 피처
    numerical_feat = [
        'last_charge_end_time_ts',
        'connection_start_time_ts',
        'charging_start_time_ts',
        'charging_end_time_ts',
        'connection_end_time_ts',
        'expected_departure_time_ts',
        'idle_time_ts',
        'expected_usage_duration_ts',
        'expected_time_diff_ts',
        'actual_usage_duration_ts',
        'actual_charging_duration_ts',
        'start_delay_duration_ts',
        'post_charge_departure_delay_ts',
        'usage_departure_time_diff_ts',
        'duration_per_kwh_ts',
        'delivered_kwh',
        'requested_kwh',
        'kwh_request_diff',
        'kwh_per_usage_time'
    ]
    categorical_feat = [
    'charging_start_time_missing',
    'charging_end_time_missing',
    'expected_departure_time_missing',
    'expected_usage_duration_missing',
    'expected_time_diff_missing',
    'actual_charging_duration_missing',
    'start_delay_duration_missing',
    'post_charge_departure_delay_missing',
    'usage_departure_time_diff_missing',
    'kwh_per_usage_time_missing',
    'evse_type',
    'supports_discharge',
    'scheduled_charge'
    ]

    numerical_transformer = Pipeline([('imputer', SimpleImputer(strategy="median")),
                                    ('scaler', StandardScaler())])

    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")),
                                        ('onehot', OneHotEncoder(sparse_output=True, handle_unknown="ignore"))])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_feat),
        ('cat', categorical_transformer, categorical_feat),
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    return model
