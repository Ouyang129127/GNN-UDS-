# GNN-UDS项目 main.py 代码结构详细解析

## 项目背景

**GNN-UDS** (Graph Neural Network-based Urban Drainage Surrogate) 是一个用于城市排水网络的液压替代模型和实时控制方法。该项目基于TensorFlow开发，旨在通过图神经网络(GNN)构建排水系统的替代模型，实现实时水力预测和模型预测控制(MPC)。

## 论文引用

1. **GNN-based model**: Zhang, Z., Tian, W., Lu, C., Liao, Z. and Yuan, Z. 2024. Graph neural network-based surrogate modelling for real-time hydraulic prediction of urban drainage networks. Water Research, 263, 122142.
2. **Gradient-based MPC**: Zhang, Z., Tian, W., Liao, Z. and Yuan, Z. 2026. Differentiable neural network-based models enable gradient-based optimization for model predictive control of urban drainage networks. Water Research, 291, 125188.

## main.py 代码结构总览

`main.py` 是GNN-UDS项目的核心入口文件，实现了整个工作流程的四个主要阶段：
1. **数据生成** (Simulation)
2. **模型训练** (Training)
3. **模型验证** (Validation)
4. **模型测试** (Testing)

## 一、导入模块

```python
import tensorflow as tf
tf.config.list_physical_devices(device_type='GPU')  # GPU设备检测
from emulator import Emulator  # 替代模型核心模块
from dataloader import DataGenerator  # 数据生成器
from utils.utilities import get_inp_files  # 文件处理工具
import argparse, yaml  # 命令行参数解析和YAML配置
from envs import get_env  # 环境获取函数
import numpy as np
import os, time, shutil
import pandas as pd
import multiprocessing as mp  # 多进程支持
from mpc import get_runoff, pred_simu  # MPC相关函数
import matplotlib.pyplot as plt
from keras.utils import plot_model  # 模型可视化

# 路径定义
HERE = os.path.dirname(__file__)

# 混合精度设置
from keras import mixed_precision
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)
```

## 二、命令行参数解析类 `Argument`

### 2.1 环境参数 (Environment Args)
- `--env`: 排水场景选择 (`shunqing`, `astlingen`, `hague`, `chaohu`, `RedChicoSur`)
- `--directed`: 是否使用有向图
- `--length`: 邻接范围
- `--order`: 邻接阶数
- `--graph_base`: 图结构基础（0:默认，1:节点，2:边）
- `--rain_dir`: 降雨事件路径
- `--rain_suffix`: 降雨文件后缀
- `--rain_num`: 降雨事件数量
- `--swmm_step`: SWMM模拟步长

### 2.2 模拟参数 (Simulate Args)
- `--simulate`: 是否生成训练数据
- `--data_dir`: 采样数据文件路径
- `--train_event_id`: 训练事件ID文件
- `--act`: 控制动作类型（`False`无控制，`conti`连续控制，`rand`随机控制）
- `--ctrl_step`: 控制步长
- `--processes`: 模拟进程数
- `--repeats`: 每个事件的模拟重复次数

### 2.3 训练参数 (Train Args)
- `--train`: 是否训练替代模型
- `--seed`: 随机种子
- `--load_model`: 是否加载现有模型继续训练
- `--edge_fusion`: 是否使用节点-边融合模型
- `--use_adj`: 是否使用滤波器进行控制
- `--model_dir`: 模型权重保存路径
- `--ratio`: 训练事件比例
- `--learning_rate`: 学习率
- `--epochs`: 训练轮数
- `--save_gap`: 模型保存间隔
- `--batch_size`: 批次大小
- `--roll`: 是否使用课程学习展开
- `--balance`: 是否使用平衡损失而非分类损失
- `--gradnorm`: 是否使用GradNorm平衡多任务学习

### 2.4 网络参数 (Network Args)
- `--conv`: 卷积类型 (`GCNconv`, `GATconv`, `ChebConv`)
- `--embed_size`: 卷积层通道数
- `--n_sp_layer`: 空间层数
- `--dropout`: Dropout率
- `--activation`: 激活函数
- `--recurrent`: 循环类型 (`GRU`, `LSTM`, `Conv1D`)
- `--hidden_dim`: 循环层通道数
- `--kernel_size`: 卷积核大小
- `--n_tp_layer`: 时序层数
- `--seq_in`: 输入序列长度
- `--seq_out`: 输出序列长度（需小于输入序列）
- `--resnet`: 是否使用残差网络
- `--if_flood`: 是否进行洪水分类
- `--epsilon`: 洪水深度阈值

### 2.5 验证参数 (Validate Args)
- `--validate`: 是否验证替代模型
- `--horizon`: 预测时域长度
- `--pop_size`: 并行控制选项数量

### 2.6 测试参数 (Test Args)
- `--test`: 是否测试替代模型
- `--hotstart`: 是否使用热启动测试模拟时间

## 三、主要函数解析

### 3.1 `parser(config=None)` 函数
**功能**: 解析命令行参数和配置文件
**流程**:
1. 创建`Argument`解析器
2. 解析命令行参数
3. 如果提供配置文件，加载YAML配置并更新参数
4. 处理路径相关参数的拼接
5. 返回参数对象和配置字典

### 3.2 主程序流程 (`if __name__ == "__main__":`)

#### 阶段一：初始化设置
1. 解析命令行参数
2. 设置随机种子确保结果可复现
3. 获取环境实例
4. 获取环境参数并更新args
5. 初始化数据生成器

#### 阶段二：数据生成 (`if args.simulate:`)
1. 创建数据目录
2. 保存配置到YAML文件
3. 获取降雨事件文件
4. 调用数据生成器生成训练数据
5. 保存生成的数据

#### 阶段三：模型训练 (`if args.train:`)
1. 加载训练数据
2. 创建模型目录
3. 生成训练/测试数据集划分
4. 初始化替代模型(`Emulator`)
5. 加载预训练模型（如果设置）
6. 设置数据归一化参数
7. 保存配置到YAML文件
8. **训练循环**:
   - 准备训练批次数据
   - 数据归一化
   - 模型训练和评估
   - 保存最佳模型
   - 记录损失和性能指标
   - TensorBoard日志记录
9. 保存最终模型、训练ID、损失曲线和时间记录

#### 阶段四：模型验证 (`if args.validate:`)
1. 加载已知的模型配置
2. 更新环境参数
3. 加载训练好的替代模型
4. 创建结果目录
5. 获取降雨事件文件
6. **对每个降雨事件**:
   - 获取径流数据
   - 初始化环境状态
   - 执行模型预测控制仿真
   - 比较替代模型和物理模型的预测结果
   - 记录性能指标
7. 保存所有验证结果

#### 阶段五：模型测试 (`if args.test:`)
1. 加载已知的模型配置
2. 更新环境参数
3. 初始化数据生成器
4. 加载训练好的替代模型
5. 创建结果目录
6. **对每个降雨事件**:
   - 模拟或加载物理模型的运行数据
   - 准备测试数据序列
   - 使用替代模型进行预测
   - 计算预测误差
   - 保存预测结果和真实值
   - 输出测试损失统计

## 四、核心数据结构流程

### 训练阶段数据流
```
降雨事件 → 物理模拟 → 状态数据 → 数据生成器 → 训练批次 → 替代模型 → 损失计算 → 模型更新
```

### 验证阶段数据流
```
降雨事件 → 环境初始化 → 状态提取 → 替代模型预测 → 性能评估 → 结果保存
```

### 测试阶段数据流
```
降雨事件 → 物理模拟数据 → 序列化处理 → 替代模型预测 → 误差计算 → 性能分析
```

## 五、关键技术要点

### 5.1 图神经网络架构
- 空间卷积层：处理排水网络的图结构
- 时序循环层：处理时间序列依赖
- 多任务学习：同时预测节点状态和边状态
- 洪水分类：作为辅助任务提升模型性能

### 5.2 数据归一化
```python
x,b,y,ex,ey = [emul.normalize(dat,item) for dat,item in zip([x,b,y,ex,ey],'xbyee')]
```
- `'x'`: 节点状态
- `'b'`: 边界条件
- `'y'`: 节点预测目标
- `'e'`: 边状态

### 5.3 多进程支持
```python
pool = mp.Pool(args.processes)
res = [pool.apply_async(func=pred_simu,args=(sett,eval_file,args,ri,...))]
```
用于并行模拟多个控制选项，提高计算效率。

### 5.4 损失函数
- **节点损失**: 均方误差(MSE)用于节点状态预测
- **边损失**: MSE用于边状态预测
- **洪水分类损失**: 二元交叉熵(BCE)用于洪水分类

### 5.5 模型保存与加载
- 使用TensorFlow SavedModel格式
- 保存最佳训练和测试模型
- 支持从断点继续训练

## 六、支持的排水网络场景

1. **shunqing**: 暴雨排水网络 (113节点, 131管道)
2. **astlingen**: 合流制排水网络 (30节点, 29边)
3. **chaohu**: 合流制排水网络 (2个泵站)
4. **hague**: 暴雨排水网络 (2个调蓄池)
5. **RedChicoSur**: 小型排水网络案例

## 七、运行示例

### 7.1 生成训练数据
```bash
python main.py --simulate --env shunqing --data_dir ./data/shunqing/ --act conti
```

### 7.2 训练模型
```bash
python main.py --train --env shunqing --data_dir ./data/shunqing/ --model_dir ./model/shunqing/ --edge_fusion --act conti --conv GAT --epochs 5000
```

### 7.3 测试模型
```bash
python main.py --test --env shunqing --model_dir ./model/shunqing/ --rain_dir ./envs/config/shunqing_events.csv --result_dir ./results/shunqing/
```

## 八、项目依赖
- TensorFlow 2.10.0
- Keras 2.10.0
- Spektral 1.3.1 (图神经网络库)
- PySWMM 1.5.1 (排水模拟)
- pystorms 1.0.0
- PyMoo 0.6.0 (多目标优化)

## 九、代码优化建议

1. **模块化改进**:
   - 将四个主要阶段拆分为独立模块
   - 增加配置文件验证机制
   - 添加异常处理和日志记录

2. **性能优化**:
   - 使用TensorFlow Dataset API改进数据流水线
   - 实现混合精度训练(代码中已有设置但未完全使用)
   - 添加分布式训练支持

3. **功能扩展**:
   - 支持更多类型的图神经网络架构
   - 添加模型压缩和部署功能
   - 集成在线学习功能

4. **可维护性**:
   - 增加单元测试和集成测试
   - 添加类型提示
   - 完善文档字符串

## 十、总结

`main.py` 展现了深度学习在环境工程领域的创新应用，通过图神经网络构建排水系统的"数字孪生"，实现了：
1. **高效模拟替代**: 比传统物理模型快几个数量级
2. **实时预测**: 支持实时水力状态预测
3. **控制优化**: 为模型预测控制提供可微分的内部模型
4. **场景适应**: 支持多种排水网络配置

该代码结构清晰，功能完整，为城市排水系统的智能化管理和优化提供了强有力的技术工具。