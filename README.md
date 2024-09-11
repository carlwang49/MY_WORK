# Multi-Agent RL for Cost-effective Bidirectional EV Charging Strategies (基於多智能體強化學習的電動車充放電成本優化策略)

## Code Flow

### Preprocessing
* `LSTM_fill.ipynb`: 用於數據填充的 LSTM 模型。
* `DataPreprocessing.ipynb`: 用於數據的預處理工作。這個 Jupyter Notebook 會對原始數據進行清理、擴充和格式化，以便生成模擬過程所需的資料集。

### Model Training and Algorithms
#### DDPG
* `DDPG/`: 包含深度確定性策略梯度（DDPG）算法的實現，用於電動車充電站和建築能源管理中的決策優化。
  * `ActionSpace.py`: 定義動作空間，確定在不同情況下可以採取的行動。
  * `DDPG.py`: DDPG 算法的核心實現。
  * `EVBuildingEnv.py`: 建築能源管理的環境定義。
  * `EVChargingEnv.py`: 電動車充電環境的定義。
  * `NetWork.py`: 定義神經網絡架構，用於策略和價值函數近似。
  * `PriceEnvironment.py`: 價格環境定義，用於模擬不同的電力價格情境。
  * `logger_config.py`: 日誌設置文件。
  * `main.py`: 執行DDPG算法的主文件。
  * `utils.py`: 通用工具函數，用於支持 DDPG 的運行。

#### GB-MARL (our method)
* `GB-MARL/`: 提供基於分層的多智能體強化學習（MARL）算法，主要針對電動車充電和放電的場景。
  * `ActionSpace.py`: 定義智能體的動作空間。
  * `Agent.py`: 單個智能體的定義及其行為策略。
  * `Buffer.py`: 經驗重放緩衝區的實現。
  * `ChargingAgent.py`: 充電智能體的實現。
  * `DischargingAgent.py`: 放電智能體的實現。
  * `EVBuildingEnvGB_MARL.py`: 建築能源管理環境的實現，支持多智能體協作。
  * `EVChargingEnv.py`: 電動車充電環境的實現。
  * `GB_MARL.py`: 基於圖形的 MARL 算法的主體實現。
  * `logger_config.py`: 日誌設置文件。
  * `main.py`: 執行 MARL 算法的主文件。
  * `utils.py`: 通用工具函數，用於支持 MARL 的運行。

#### LinearProgrammingDayAhead
* `LinearProgrammingDayAhead/`: 基於線性規劃的日內調度算法，用於優化建築能源管理。
  * `DayAheadSchedule.py`: 核心日內調度算法的實現。
  * `DayAheadScheduleOneDay.py`: 單日調度的算法實現。
  * `logger_config.py`: 日誌設置文件。
  * `main.py`: 執行日內調度算法的主文件。
  * `utils.py`: 通用工具函數，用於支持調度算法的運行。


#### MultiAgentDDPG
* `MultiAgentDDPG/`: 多智能體版的 DDPG 算法實現。
  * `ActionSpace.py`: 定義多智能體的動作空間。
  * `Agent.py`: 單個智能體的定義及其策略。
  * `Agent_ParameterSharing.py`: 使用參數共享的多智能體實現。
  * `Buffer.py`: 經驗重放緩衝區的實現。
  * `EVBuildingEnvMADDPG.py`: 為多智能體 DDPG 設計的建築環境管理系統。
  * `EVChargingEnv.py`: 電動車充電環境的實現。
  * `MADDPG.py`: 多智能體 DDPG 算法的主體實現。
  * `MADDPG_ParameterSharing.py`: 使用參數共享的多智能體 DDPG 實現。
  * `PriceEnvironment.py`: 價格環境定義，用於模擬不同的電力價格情境。
  * `evaluate.py`: 評估多智能體 DDPG 算法的性能。
  * `logger_config.py`: 日誌設置文件。
  * `maddpg_parameter.py`: 參數設置文件，用於調整 DDPG 算法的運行參數。
  * `main.py`: 執行 DDPG 算法的主文件。
  * `utils.py`: 通用工具函數，用於支持 DDPG 的運行。

#### MultiAgentIQL
* `MultiAgentIQL/`: 使用獨立 Q-Learning（IQL）算法的多智能體強化學習實現。
  * `ActionSpace.py`: 定義多智能體的動作空間。
  * `EVBuildingEnv.py`: 為多智能體 IQL 設計的建築環境管理系統。
  * `EVChargingEnv.py`: 電動車充電環境的實現。
  * `PriceEnvironment.py`: 價格環境定義，用於模擬不同的電力價格情境。
  * `QLearningAgent.py`: 獨立 Q-Learning 智能體的實現。
  * `trained_agents/`: 已訓練的 IQL 智能體。
  * `utilities.py`: 通用工具函數。
  * `utils/plot_results.py`: 結果可視化工具。

#### MultiAgentVDN
* `MultiAgentVDN/`: 基於價值分解網絡（Value Decomposition Networks, VDN）的多智能體強化學習策略。
  * `EVBuildingEnv.py`: 為多智能體 VDN 設計的建築環境管理系統。
  * `EVChargingEnv.py`: 電動車充電環境的實現。
  * `QNet.py`: 用於 VDN 的 Q 網絡結構。
  * `ReplayBuffer.py`: 經驗重放緩衝區的實現。
  * `agent.py`: VDN 智能體的實現。
  * `logger_config.py`: 日誌設置文件。
  * `main.py`: 執行 VDN 算法的主文件。
  * `trained_agents/`: 已訓練的 VDN 智能體。
  * `utilities.py`: 通用工具函數。
  * `utils/plot_results.py`: 結果可視化工具。

### Evaluation
* `Evaluation-v2.ipynb`: 用於評估模型性能的 Jupyter Notebook，包含多個基準和算法的比較結果。
* `Evaluation.ipynb`: 另一個評估模型的 Jupyter Notebook，提供不同的評估視角。

## Installation

### Requirements
請確保您的環境滿足以下要求：

1. 安裝 `requirements.txt` 中列出的 Python 依賴包：

    ```bash
    pip install -r requirements.txt
    ```
2. 下載 Dataset: https://drive.google.com/file/d/1yVtRbo8OIYw0s1ZjhvCjQpNXeko2L7Uo/view?usp=sharing

## Usage

1. **數據預處理**: 使用 `DataPreprocessing.ipynb` 來清理和準備訓練數據。

2. **新增和設定環境檔**: 新增 .env 檔，設定範例如下:
  
  ```
  # Description: Environment variables for the EV charging system
  # Simulation settings
  START_DATETIME=2018-07-01 # Start date of the simulation
  END_DATETIME=2018-08-01 # End date of the simulation
  TEST_START_DATETIME=2018-08-01 # Test date of the simulation
  TEST_END_DATETIME=2018-09-01 # Test date of the simulation
  NUM_AGENTS=10  # Number of charging piles
  # DIR_NAME=GB-MARL # Directory name for saving the results

  # EV settings
  BATTERY_CAPACITY=60 # Maximum battery capacity (kWh)
  SOC_MIN=0.2 # Minimum state of charge (SoC) for EVs
  SOC_MAX=0.9 # Maximum state of charge (SoC) for EVs
  MAX_CHARGING_POWER=150 # Rated maximum charging power of charging piles (kW)
  MAX_DISCHARGING_POWER=-150 # Rated maximum discharging power of charging piles (kW)
  ENERGY_CONVERSION_EFFICIENCY=0.98 # Coefficient of energy conversion efficiency

  # Building settings
  MAX_LOAD=1000 # Maximum load of the building (kW)
  MIN_LOAD=0 # Minimum load of the building (kW)
  CONTRACT_CAPACITY=700 # Contract capacity of the building (kW)
  CAPACITY_PRICE=15 # Capacity price of the building ($/kW)

  # Parameters for the reward function
  REWARD_ALPHA = 0.5 # Weight
  REWARD_BETA = 0.5 # Weight
  REWARD_GAMMA = 0.5 # Weight

  # Hyperparameters
  NUMBER_OF_EPISODES=3000 # Number of total training episodes
  LEARN_INTERVAL=100 # Interval for training the actor and critic networks
  RANDOM_STEPS=5e4 # Number of random steps before training
  TAU=0.01 # Tau for soft update of target networks
  GAMMA=0.97 # Discount factor for future rewards
  AGENT_BUFFER_CAPACITY=3e4 # Size of the replay buffer
  TOP_LEVEL_BUFFER_CAPACITY=3e4 # Size of the top-level replay buffer
  AGENT_BATCH_SIZE=1024 # Mini-batch size for training the agent networks
  TOP_LEVEL_BATCH_SIZE=1024 # Mini-batch size for training the top-level network
  LEARNING_RATE_ACTOR=0.00005 # Learning rate for the actor network
  LEARNING_RATE_CRITIC=0.00005 # Learning rate for the critic network
  EPSILON = 0.9 # Epsilon for the epsilon-greedy policy
  SIGMA = 0.1 # Sigma for the Ornstein-Uhlenbeck noise
  SIGMA_DECAY = 0.995 # Sigma decay for the Ornstein-Uhlenbeck noise
  ```
   
3. **訓練模型**: 在各子模組的 `main.py` 文件中運行相應的算法來進行模型訓練。

   ```bash
   python3 main.py
