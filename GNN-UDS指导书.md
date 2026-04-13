# GNN-UDS 城市排水系统模拟与控制平台使用指南

## 项目概述

GNN-UDS（Graph Neural Network for Urban Drainage Systems）是一个基于图神经网络的城市排水系统智能模拟与控制平台。项目结合SWMM模型、强化学习技术和GNN神经网络，实现对城市排水系统的高效模拟、预测和优化控制。

**核心特点：**
- GNN图神经网络处理排水系统拓扑结构：
- 基于SWMM和pystorms的物理模型耦合
- 强化学习（SAC、PPO、QMIX等）算法实现智能控制
- 多智能体协同控制系统优化
- 支持5个真实排水系统案例

## 系统架构

### 项目结构
```
GNN-UDS-Repo/
├── surrogate/          # 核心代码
│   ├── main.py        # 主程序入口
│   ├── emulator.py    # 环境模拟器
│   ├── predictor.py   # GNN预测模型
│   ├── dataloader.py  # 数据加载器
│   ├── mpc.py         # 模型预测控制
│   ├── agent.py       # 强化学习智能体
│   ├── mbrl.py        # 基于模型的强化学习
│   ├── maxred.py      # 最大化减排算法
│   └── utilsc/         # 工具模块
├── envs/              # 环境配置文件
├── model/             # 训练模型存储
├── results/           # 实验结果
└── requirements.txt   # 依赖包列表
```

### 数据处理流程
```
雨水数据 → SWMM模拟 → GNN训练 → 强化学习优化 → 实时控制
输入降水 → 物理模型 → 代理模型 → 智能决策 → 排放口控制
```

## 环境配置与安装

### 1. Python环境要求
- Python 3.8+
- TensorFlow 2.12.0（已验证兼容性）
- 建议使用虚拟环境（conda或venv）

### 2. 安装步骤
```bash
# 克隆项目
git clone https://github.com/Zhiyu014/GNN-UDS.git
cd GNN-UDS

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows   （第一个问题，对于项目环境配置前创建虚拟环境的意义与必要性）

# 安装依赖
pip install -r requirements.txt

# 注意：TensorFlow需单独处理
pip install tensorflow==2.12.0
```

### 3. 所需依赖包
```
核心依赖:
- tensorflow==2.12.0        # 深度学习框架
- spektral                  # 图神经网络库
- pyswmm==1.5.0            # SWMM Python接口
- pystorms                  # 排水系统模拟库
- numpy>=1.19.0
- scipy>=1.7.0
- pandas>=1.3.0

高级算法（已集成）:
- torch                    # PyTorch（可选）
- stable-baselines3        # 强化学习库
- ray[rllib]              # 分布式强化学习
```

## 核心功能模块详解

### 1. 环境模拟器 (emulator.py)
**功能：** 基于SWMM的排水系统物理模型模拟器

**主要参数：**
```python
class UrbanDrainageEnv:
    def __init__(self, config):
        self.network_graph   # 排水系统图结构
        self.rain_events     # 降雨事件
        self.control_nodes   # 可控排放口
        self.swmm_model      # SWMM模型实例
        self.step_size       # 模拟步长(1-60秒)
```

**使用方法：**
```python
from surrogate.emulator import UrbanDrainageEnv

# 创建环境
env = UrbanDrainageEnv(
    network="shunqing",      # 排水网络
    rain_file="rain_events.csv",
    step_size=5,            # 5秒步长
    simulation_days=10
)

# 运行模拟
obs, reward, done, info = env.step(actions)
```

### 2. GNN预测模型 (predictor.py)
**功能：** 图神经网络对排水系统的状态预测

**模型架构：**
```
输入层（节点特征） → GAT/GCN层 → GRU时间序列层 → 输出层
    ↓                     ↓               ↓           ↓
降雨量+水位 → 图注意力机制 → 时间记忆 → 预测结果
    ↓                     ↓               ↓           ↓
管网拓扑 → 邻接矩阵 → 隐藏状态 → 水位/流量预测
```

**关键技术：**
- **GATconv**: 图注意力卷积层，关注邻接节点重要性
- **GRU**: 门控循环单元，处理时间序列依赖性
- **ResNet连接**: 残差网络防止梯度消失
- **Edge Fusion**: 边特征融合，管道特征建模

**训练配置：**
```yaml
network:
  conv: GATconv          # 图卷积类型
  embed_size: 128        # 特征维度
  n_sp_layer: 2         # 空间层数
  recurrent: GRU        # 时序网络类型
  hidden_dim: 64        # 隐藏维度
  seq_in: 6            # 输入序列长度
  seq_out: 1           # 输出序列长度
```

### 3. 强化学习智能体 (agent.py)
**功能：** 训练智能体优化排水系统控制策略

**支持的算法：**
- **SAC**: Soft Actor-Critic（连续动作空间）
- **PPO**: Proximal Policy Optimization
- **TD3**: Twin Delayed DDPG
- **QMIX**: 多智能体混合网络
- **MADDPG**: 多智能体DDPG

**智能体配置：**
```python
from surrogate.agent import RLAgent

agent = RLAgent(
    algorithm="sac",        # 算法类型
    action_space=continuous,# 动作空间
    observation_dim=64,     # 观测维度
    memory_size=10000,      # 经验池大小
    batch_size=256,        # 训练批大小
    gamma=0.99,           # 折扣因子
    tau=0.005             # 软更新系数
)
```

### 4. 数据加载器 (dataloader.py)
**功能：** 处理SWMM模拟数据，构建GNN数据集

**数据流程：**
```
SWMM模拟输出 → 数据清理 → 拓扑构建 → 时间序列切片 → GNN数据集
    ↓              ↓           ↓            ↓            ↓
.inp/.out文件 → 特征归一化 → 邻接矩阵 → 滑动窗口 → 训练/验证/测试集
```

**支持的排水系统：**
1. **Shunqing**（顺庆）：中国四川排水系统
2. **Astlingen**：德国排水系统
3. **Hague**（海牙）：荷兰排水系统
4. **RedChicoSur**：智利排水系统
5. **Chaohu**（巢湖）：中国合肥排水系统

## 配置文件详解

### config.yaml 配置说明
位于 `surrogate/utils/config.yaml`

**主要配置部分：**
```yaml
# 环境参数
shunqing:
  env: shunqing
  directed: False        # 是否定向图
  rain_num: 119         # 降雨事件数量
  simulate: False       # 是否运行模拟
  
# 模拟参数
simulate: False
data_dir: ./envs/data/shunqing/
processes: 5           # 并行进程数
repeats: 1            # 重复模拟次数

# 训练参数
train: False
seed: 42              # 随机种子
learning_rate: 0.001
epochs: 500
batch_size: 256
gradnorm: False       # 梯度归一化

# 神经网络参数
conv: GATconv
embed_size: 128
hidden_dim: 64
seq_in: 6            # 输入6个时间步
seq_out: 1           # 预测1个时间步
```

## 操作流程

### 第一阶段：数据准备（1-2周）
**步骤1：环境初始化**
```bash
cd D:\DeepLearning_Model\projects\GNN-UDS\GNN-UDS-Repo\surrogate
py main.py --env shunqing --simulate True
```

**步骤2：运行SWMM模拟**
- 自动生成降雨数据
- 运行排水系统物理模拟
- 保存模拟结果到 `./envs/data/shunqing/`
- 生成: `train_id_X.npy`（训练数据标识）

**步骤3：数据处理**
- GNN图结构构建（节点特征、邻接矩阵）
- 时间序列标准化
- 数据集分割（训练80%，验证10%，测试10%）

### 第二阶段：模型训练（2-3周）
**步骤4：GNN预测器训练**
```bash
py main.py --env shunqing --train True --model_dir ./model/shunqing/
```

**训练过程：**
- 加载处理后的数据
- 训练GNN时间序列预测器
- 验证集调优，防止过拟合
- 保存最佳模型权重

**步骤5：强化学习智能体训练**
```bash
py main.py --env shunqing --train True --algorithm sac
# 或使用预训练命令
py test.bat
```

**智能体训练模式：**
- **MBRL**（Model-Based RL）：基于环境模型
- **MFRL**（Model-Free RL）：不依赖模型
- **协作训练**：多个排放口协同控制

### 第三阶段：测试与评估（1周）
**步骤6：性能评估**
```bash
py main.py --env shunqing --test True --result_dir ./results/shunqing/
```

**评估指标：**
1. **预测精度**: MAE, RMSE, R²
2. **控制效果**: 内涝体积、溢流频率、峰值流量
3. **计算效率**: 模拟速度vs物理模型
4. **鲁棒性**: 不同降雨事件下的稳定性

**步骤7：可视化分析**
- 生成降雨-水位-控制曲线
- 节点流量分布热力图
- 训练损失收敛曲线
- 控制决策时间序列

## 高级应用

### 1. 定制化降雨场景
```python
# 创建自定义降雨事件
import numpy as np
from utils.utilities import create_rain_events

rain_pattern = np.array([...])  # 降雨模式
events = create_rain_events(
    pattern=rain_pattern,
    duration=24,        # 24小时
    intensity_scale=1.5 # 1.5倍强度
)
```

### 2. 多目标优化
```python
# 配置多目标优化
from surrogate.mpc import MultiObjectiveMPC

mpc = MultiObjectiveMPC(
    objectives=[
        "minimize_flood_volume",     # 最小化内涝体积
        "maximize_flow_capacity",    # 最大化排水能力
        "minimize_control_inputs"    # 最小化控制动作
    ],
    weights=[0.5, 0.3, 0.2]          # 目标权重
)
```

### 3. 实时控制接口
```python
# 实时控制API
from surrogate.maxred import RealTimeController

controller = RealTimeController(
    model_path="./model/shunqing/best_model.h5",
    control_horizon=60,      # 60分钟控制视野
    update_interval=5        # 5分钟更新一次
)

# 获取实时控制指令
while True:
    rain_forecast = get_rain_forecast()
    system_state = get_system_status()
    control_action = controller.step(rain_forecast, system_state)
    
    # 执行控制（如阀门开度调节）
    execute_control(control_action)
    time.sleep(5 * 60)  # 等待5分钟
```

## 典型应用场景

### 应用1：暴雨内涝预警系统
```
降雨预报 → GNN预测 → 风险评估 → 预警发布
    ↓          ↓          ↓          ↓
气象数据 → 水位预测 → 内涝概率 → 分级预警
```

**代码实现：**
```python
def flood_warning_system(rain_forecast):
    # 1. GNN预测未来2小时水位
    water_levels = gnn_predictor.predict(rain_forecast)
    
    # 2. 计算内涝风险指数
    risk_scores = calculate_flood_risk(water_levels)
    
    # 3. 生成预警等级
    warning_level = classify_warning(risk_scores)
    
    # 4. 自动响应措施
    if warning_level >= 3:  # 橙色预警
        activate_emergency_pumping()
        notify_authorities()
        
    return warning_level, control_actions
```

### 应用2：智能泵站调度
```
实时监测 → 需求预测 → 优化调度 → 泵站控制
    ↓          ↓          ↓          ↓
流量数据 → ML预测 → 强化学习 → 变频调节
```

### 应用3：污染扩散模拟
```
降雨径流 → 污染物传输 → GNN扩散模型 → 影响评估
    ↓           ↓              ↓             ↓
CSO事件 → 物质传输 → 拓扑传播 → 风险地图
```

## 性能优化建议

### 1. 硬件配置要求
- **最低配置**: 16GB RAM, 8核CPU, RTX 3060 GPU
- **推荐配置**: 32GB RAM, 12核CPU, RTX 4090 GPU
- **存储要求**: 100GB SSD用于数据存储

### 2. 计算性能优化
```python
# 启用GPU加速
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
# 使用并行处理
from multiprocessing import Pool
with Pool(processes=5) as pool:
    results = pool.map(parallel_simulation, rain_events)
```

### 3. 内存管理技巧
- 使用生成器而非一次性加载所有数据
- 定期清理TensorFlow会话
- 使用混合精度训练（FP16）
- 实施梯度累积减少批次内存

## 故障排除

### 常见问题1：TensorFlow版本冲突
```
错误：No module named 'tensorflow.python.keras'
解决：pip install tensorflow==2.12.0
```

### 常见问题2：SWMM依赖错误
```
错误：pyswmm SWMM library not found
解决：安装EPANET-SWMM或配置系统PATH
```

### 常见问题3：GPU内存不足
```
错误：CUDA out of memory
解决：
1. 减小batch_size（如从256降到64）
2. 使用gradient accumulation
3. 清理GPU缓存：tf.keras.backend.clear_session()
```

### 常见问题4：强化学习不收敛
```
现象：reward不上升或震荡
解决：
1. 调整learning_rate（如0.001→0.0001）
2. 增加经验回放池大小
3. 添加探索噪声衰减
4. 使用更稳定的算法（PPO替代SAC）
```

## 研究扩展方向

### 1. 算法改进方向
- **图结构学习**: 自动学习排水系统拓扑
- **迁移学习**: 跨城市模型迁移
- **在线学习**: 实时更新模型参数
- **联邦学习**: 多城市协同训练

### 2. 应用扩展方向
- **耦合气象预报**: 集成WRF等气象模型
- **社会经济影响评估**: 内涝损失模型
- **多尺度建模**: 街区-城市-流域多尺度
- **数字孪生系统**: 实时监控+预测+控制一体化

### 3. 论文发表方向
基于本项目可延伸的学术研究方向：
1. **期刊论文**: Water Research, Environmental Science & Technology
2. **会议论文**: NeurIPS, ICML, ICLR（机器学习）
3. **领域会议**: WDSA, World Water Congress
4. **申请专利**: 智能排水控制算法专利

## 快速启动指南

### 第一步：环境验证（5分钟）
```bash
cd D:\DeepLearning_Model\projects\GNN-UDS\GNN-UDS-Repo\surrogate
py -c "import tensorflow as tf; print('TF:', tf.__version__)"
py -c "import pyswmm; print('SWMM OK')"
py verify_env.py
```

### 第二步：简单测试（15分钟）
```bash
# 运行测试脚本
py test.bat
# 或手动测试
py main.py --env shunqing --train False --test True
```

### 第三步：完整流程（建议分配时间）
```
1. 模拟数据生成：3小时
2. GNN模型训练：12小时
3. RL智能体训练：24-48小时
4. 测试评估：3小时
5. 可视化分析：2小时
```

### 第四步：结果查看
```
./results/shunqing/
├── test_predictions.csv     # 预测结果
├── control_actions.csv      # 控制动作
├── performance_metrics.json # 评估指标
├── training_loss.png        # 损失曲线
└── control_timeline.png     # 控制时间线
```

## 补充资源

### 1. 项目文档
- **原始论文**: arXiv:2303.xxxxx [cs.LG]
- **技术手册**: ./docs/technical_manual.pdf（存在时）
- **API文档**: 查看源代码中的docstring

### 2. 数据资源
- **降雨数据**: NOAA, ECMWF, 中国气象局
- **排水系统**: EPA SWMM案例库, pystorms
- **地形数据**: USGS, OpenStreetMap, 高德地图API

### 3. 相关工具
- **SWMM可视**: PCSWMM, SWMM5 GUI
- **GIS集成**: ArcGIS, QGIS的SWMM插件
- **实时监控**: Grafana, Prometheus数据看板
- **云部署**: Docker容器化部署

---

## 更新记录

### **v1.0**（当前已部署）
- 基础GNN预测模型
- 5个真实排水系统
- 多种RL算法支持
- TensorFlow 2.12.0兼容性

### **v1.1**（规划中）
- 支持更多降雨模式
- 在线学习功能
- GUI控制界面
- 云API接口

---

**技术支持：**
- **项目维护**: Zhiyu014 (GitHub)
- **环境问题**: 参照requirements.txt版本
- **科学咨询**: 排水系统建模专家
- **程序错误**: 提交GitHub Issue

**注意：** 本文档为项目使用指南，实际操作请结合具体数据和硬件条件进行调整。