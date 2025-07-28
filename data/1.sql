ALTER TABLE statistics
ADD COLUMN id INT NOT NULL AUTO_INCREMENT PRIMARY KEY;

SELECT 
  station_location,
  DAYNAME(FROM_UNIXTIME(charging_start_time_ts)) AS 요일,
  AVG(requested_kwh) AS 평균_requested_kwh
FROM `store_time_data`
WHERE 
  station_location like 'CSCS2015' and
  DAYOFWEEK(FROM_UNIXTIME(charging_start_time_ts)) BETWEEN 1 AND 7 -- 월=2 ~ 금=6
GROUP BY station_location,요일;
