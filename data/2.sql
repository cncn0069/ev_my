CREATE TABLE store_time_data (
    id BIGINT NOT NULL AUTO_INCREMENT,
    last_charge_end_time_ts BIGINT DEFAULT NULL,
    connection_start_time_ts BIGINT DEFAULT NULL,
    charging_start_time_ts DOUBLE DEFAULT NULL,
    charging_start_time_missing BOOLEAN DEFAULT NULL,
    charging_end_time_ts DOUBLE DEFAULT NULL,
    charging_end_time_missing BOOLEAN DEFAULT NULL,
    connection_end_time_ts BIGINT DEFAULT NULL,
    expected_departure_time_ts INT DEFAULT NULL,
    expected_departure_time_missing INT DEFAULT NULL,
    idle_time_ts INT DEFAULT NULL,
    expected_usage_duration_ts INT DEFAULT NULL,
    expected_usage_duration_missing BOOLEAN DEFAULT NULL,
    expected_time_diff_ts DOUBLE DEFAULT NULL,
    expected_time_diff_missing BOOLEAN DEFAULT NULL,
    actual_usage_duration_ts INT DEFAULT NULL,
    actual_charging_duration_ts DOUBLE DEFAULT NULL,
    actual_charging_duration_missing BOOLEAN DEFAULT NULL,
    start_delay_duration_ts DOUBLE DEFAULT NULL,
    start_delay_duration_missing BOOLEAN DEFAULT NULL,
    post_charge_departure_delay_ts DOUBLE DEFAULT NULL,
    post_charge_departure_delay_missing BOOLEAN DEFAULT NULL,
    usage_departure_time_diff_ts INT DEFAULT NULL,
    usage_departure_time_diff_missing BOOLEAN DEFAULT NULL,
    duration_per_kwh_ts INT DEFAULT NULL,
    duration_per_kwh_missing BOOLEAN DEFAULT NULL,
    delivered_kwh DOUBLE DEFAULT NULL,
    requested_kwh DOUBLE DEFAULT NULL,
    kwh_request_diff DOUBLE DEFAULT NULL,
    kwh_per_usage_time DOUBLE DEFAULT NULL,
    kwh_per_usage_time_missing BOOLEAN DEFAULT NULL,
    station_location VARCHAR(255) DEFAULT NULL,
    evse_name VARCHAR(255) DEFAULT NULL,
    evse_type INT DEFAULT NULL,
    supports_discharge BOOLEAN DEFAULT NULL,
    scheduled_charge BOOLEAN DEFAULT NULL,
    weekday INT DEFAULT NULL,
    usage_departure_range INT DEFAULT NULL,
    post_charge_departure_range INT DEFAULT NULL,
    cluster INT DEFAULT NULL,
    PRIMARY KEY (id)
)  ENGINE=INNODB DEFAULT CHARSET=UTF8MB4;
SHOW VARIABLES LIKE 'secure_file_priv';

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/50area_dummy_processed.csv'
INTO TABLE store_time_data
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
SET charging_start_time_ts = NULLIF(TRIM(charging_start_time_ts), ''),
    charging_start_time_ts = NULLIF(charging_start_time_ts, 'N/A'),
    charging_start_time_ts = NULLIF(charging_start_time_ts, 'null'),
    charging_start_time_ts = NULLIF(charging_start_time_ts, 'NA');


