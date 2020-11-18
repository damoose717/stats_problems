# 4. Write a function in Python that takes in a dataset (.csv file)
#    and a number of days forward, and generates predictions for hourly
#    order volume.

def forecast_volume(data_file: str, days_forward: int):
    """
    This function takes in a dataset (.csv file) and a number of days forward, 
    and generates predictions for hourly order volume.
    This prediction is based on the mean number of orders 
    for the day of week and time of day of the given timestamp.
    """

    # import libraries
    import numpy as np
    import pandas as pd
    from dateutil.relativedelta import relativedelta
    
    # read data
    df = pd.read_csv(data_file, parse_dates = ['order_hour'])
    
    # create weekday and hour columns 
    df['weekday'] = df.order_hour.dt.dayofweek
    df['hour_padded'] = df.order_hour.dt.hour.astype(str).str.pad(2, fillchar = '0')
    df['weekday_hour'] = df.weekday.astype(str) + df.hour_padded

    # get mean number of orders by day of week and hour of day
    grouped_df = df.groupby('weekday_hour').agg(np.mean)
    
    # create day of week and hour of day variable for forecast date
    df['order_hour_date'] = pd.to_datetime(df.order_hour).dt.date
    df['forecast_date'] = pd.to_datetime(
        df.order_hour_date + relativedelta(days = days_forward))
    df['forecast_order_hour'] = pd.to_datetime(
        df.forecast_date.astype(str) + ' ' + df.hour_padded + ':00:00', 
        format='%Y-%m-%d %H:%M:%S')
    df['forecast_weekday'] = df.forecast_order_hour.dt.dayofweek
    df['forecast_hour_padded'] = df.forecast_order_hour.dt.hour.astype(str).str.pad(
        2, fillchar = '0')
    df['forecast_weekday_hour'] = df.forecast_weekday.astype(str) + df.forecast_hour_padded

    # get last order's timestamp 
    last_ts = df.order_hour

    # create final dataset 
    data_out = df.loc[
        # starting at the end of the supplied dataset 
        df['forecast_order_hour'] > last_ts,  
        ['forecast_order_hour', 'forecast_weekday_hour']
    ].merge(
        grouped_df[['num_orders']], 
        left_on = 'forecast_weekday_hour', right_index = True, 
        how = 'left'
    )[['forecast_order_hour', 'num_orders']]
    data_out.columns = ['order_hour', 'predicted_order_volume']

    return data_out


