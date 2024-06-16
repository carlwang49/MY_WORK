import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 CSV 文件
csv_path = './Dataset/parking_behavior.csv' 
df = pd.read_csv(csv_path)

# 設置繪圖風格
sns.set(style="whitegrid")

# 到達時間的分佈圖
plt.figure(figsize=(10, 6))
sns.histplot(df['Arrival Time'], bins=20, kde=True)
plt.title('Distribution of Arrival Times')
plt.xlabel('Arrival Time')
plt.ylabel('Frequency')
plt.savefig('./data_plot/distribution_arrival_times.png')
plt.close()

# 離開時間的分佈圖
plt.figure(figsize=(10, 6))
sns.histplot(df['Leave Time'], bins=20, kde=True)
plt.title('Distribution of Leave Times')
plt.xlabel('Leave Time')
plt.ylabel('Frequency')
plt.savefig('./data_plot/distribution_leave_times.png')
plt.close()

# 初始 SoC 和希望的 SoC 的分佈圖
plt.figure(figsize=(10, 6))
sns.histplot(df['SoC at Arrive'], bins=20, kde=True, color='blue', label='SoC at Arrive')
sns.histplot(df['Desired SoC at Leave'], bins=20, kde=True, color='green', label='Desired SoC at Leave')
plt.title('Distribution of SoC at Arrive and Desired SoC at Leave')
plt.xlabel('State of Charge (SoC)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('./data_plot/distribution_soc.png')
plt.close()

# 停車時長的分佈圖
parking_durations = df['Leave Time'] - df['Arrival Time']
plt.figure(figsize=(10, 6))
sns.histplot(parking_durations, bins=20, kde=True)
plt.title('Distribution of Parking Durations')
plt.xlabel('Parking Duration (hours)')
plt.ylabel('Frequency')
plt.savefig('./data_plot/distribution_parking_durations.png')
plt.close()

# 到達時間與離開時間的散點圖
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Arrival Time'], y=df['Leave Time'])
plt.title('Scatter Plot of Arrival and Leave Times')
plt.xlabel('Arrival Time')
plt.ylabel('Leave Time')
plt.savefig('./data_plot/scatter_arrival_leave_times.png')
plt.close()

