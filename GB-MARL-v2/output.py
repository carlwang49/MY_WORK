import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from dotenv import load_dotenv
import seaborn as sns
from datetime import time
warnings.filterwarnings('ignore')
load_dotenv()


# calculate cost and penalty
def calculate_cost_and_penalty_v1(df, real_time_price, contract_capacity, capacity_price, start_time, end_time):
    # filter data by start and end time
    df['current_time'] = pd.to_datetime(df['current_time'])
    df_filtered = df[(df['current_time'] >= start_time) & (df['current_time'] <= end_time)].copy()
    
    # filter data by hour (7:00 - 23:00)
    df_filtered['hour'] = df_filtered['current_time'].dt.hour
    df_filtered = df_filtered[(df_filtered['hour'] >= 7) & (df_filtered['hour'] <= 23)]
    
    # merge real time price data
    df_filtered = pd.merge(df_filtered, real_time_price, how='left', left_on='current_time', right_on='datetime')

    # calculate cost
    df_filtered['cost'] = df_filtered['total_load'] * df_filtered['average_price']

    # calculate total cost
    current_tariff = df_filtered['cost'].sum()
    total_cost = current_tariff

    # calculate overload penalty
    df_filtered['month'] = df_filtered['current_time'].dt.month
    overload_penalties = []

    for month in df_filtered['month'].unique():
        monthly_data = df_filtered[df_filtered['month'] == month]
        overload_penalty = 0
        overload = monthly_data['total_load'].max()
        
        if overload > contract_capacity:
            overload -= contract_capacity
            overload_penalty += min(overload, contract_capacity * 0.1) * capacity_price * 2  # between 0% and 10% of contract capacity
            overload -= min(overload, contract_capacity * 0.1)
            overload_penalty += overload * capacity_price * 3  # more than 10% of contract capacity
        overload_penalties.append(overload_penalty)

    total_overload_penalty = sum(overload_penalties)

    return df_filtered, total_cost, total_overload_penalty, current_tariff


def set_real_time_price_range(real_time_price, start_time: datetime, end_time: datetime):
    real_time_price = real_time_price[(real_time_price['datetime'] >= start_time) & (real_time_price['datetime'] <= end_time)].copy()
    real_time_price.sort_values(by='datetime', inplace=True)
    return real_time_price


def output(result_dir):

    # set start and end datetime
    start_datetime_str = '2018-09-01'
    end_datetime_str = '2018-09-30'

    # convert string to datetime
    start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%d')

    # set contract capacity and capacity price
    contract_capacity = 700
    capacity_price = 15

    # get real time price data
    real_time_price = pd.read_csv('../Dataset/RTP/electricity_prices_from_201807010000_to_201812312359.csv')
    real_time_price['datetime'] = pd.to_datetime(real_time_price['datetime'])
    real_time_price = set_real_time_price_range(real_time_price, start_datetime, end_datetime)

    # get building loading history result data
    test_building_file = '/test_building_loading_history.csv'

    GB_MARL_dir = result_dir

    GB_MARL_building_file = GB_MARL_dir + test_building_file


    # set file paths and method names
    methods = [GB_MARL_building_file]

    method_names = ['GB-MARL']
    data_frames = []

    # read data
    for method in methods:
        df = pd.read_csv(method)
        data_frames.append(df)
        
    # calculate cost and penalty for each method
    total_costs = []
    overload_penalties = []
    current_tariffs = []
    for df, method_name in zip(data_frames, method_names):
        df_filtered, total_cost, total_overload_penalty, current_tariff = \
            calculate_cost_and_penalty_v1(df, real_time_price, contract_capacity, 
                                        capacity_price, start_datetime, end_datetime)
            
        total_cost += total_overload_penalty  # add overload penalty to total cost
        total_costs.append(total_cost)
        overload_penalties.append(total_overload_penalty)
        current_tariffs.append(current_tariff)


    # set plot style
    sns.set(style="whitegrid")

    # define colors
    colors = ['#FF6F61']

    # plot total cost bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = sns.barplot(x=method_names, y=total_costs, palette=colors, ax=ax)

    # add labels and title
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Cost (USD)', fontsize=14, fontweight='bold')
    ax.set_title('Total Electricity Cost Comparison by Method', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    # display total cost on top of each bar and print the value
    print("Total Costs by Method:")
    for bar, yval in zip(bars.patches, total_costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 * height, 
                f'${height:.4f}', ha='center', va='bottom', fontsize=12)
        print(f'{method_names[bars.patches.index(bar)]}: ${height:.4f}')
