import pandas as pd

class PriceEnvironment:
    def __init__(self, file_path, start_time, end_time):
        # Initialize the real-time electricity price
        self.real_time_price = pd.read_csv(file_path)
        self.real_time_price['datetime'] = pd.to_datetime(self.real_time_price['datetime'])
        self.real_time_price = self.set_real_time_price_range(start_time, end_time)
        self.real_time_price.sort_values(by='datetime', inplace=True)
        self.real_time_price = \
            self.real_time_price.set_index('datetime').reindex(
                pd.date_range(start=self.real_time_price['datetime'].min(), 
                              end=self.real_time_price['datetime'].max(), freq='h')).ffill().reset_index()
        self.real_time_price.rename(columns={'index': 'datetime'}, inplace=True)

        # Convert to dictionary for fast lookup
        self.price_dict = self.real_time_price.set_index('datetime')['average_price'].to_dict()

    def set_real_time_price_range(self, start_time, end_time):
        return self.real_time_price[(self.real_time_price['datetime'] >= start_time) & 
                                    (self.real_time_price['datetime'] <= end_time)]

    def get_current_price(self, timestamp):
        return self.price_dict[timestamp]
    
    def max_price(self):
        return self.real_time_price['average_price'].max()
