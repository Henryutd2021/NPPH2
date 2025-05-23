# 优化进度监控功能

本项目现在包含了基于MIP gap的智能进度监控功能，可以显示优化求解过程的真实进度。

## 功能特性

### 1. 基于Gap的进度计算

- 实时解析求解器输出，提取当前MIP gap信息
- 根据当前gap和目标gap计算真实进度百分比
- 支持Gurobi和CPLEX求解器的gap信息解析

### 2. 进度显示方式

- **有tqdm**: 显示美观的进度条，包含百分比、gap值和运行时间
- **无tqdm**: 显示简单的文字进度，包含旋转指示器、进度百分比和gap值

### 3. 支持的求解器

- **Gurobi**: 解析标准输出格式中的gap信息
- **CPLEX**: 解析日志文件中的gap信息
- **其他**: 基本的时间计数功能

## 安装依赖

为了获得最佳的进度条体验，推荐安装tqdm：

```bash
pip install tqdm
```

如果不安装tqdm，系统会自动使用备用的文字进度显示。

## 进度条样式示例

### 使用tqdm时

```
Optimizing Plant Unit 1:  45%|████▌     | Gap: 2.150% | 00:35
```

完成后会显示最终摘要：

```
Optimizing Plant Unit 1 completed in 67.3s (Final gap: 0.050%)
```

### 不使用tqdm时

```
Optimizing Plant Unit 1... | Progress: 45.0% | Gap: 2.150% | (35.0s)
```

完成后会显示：

```
Optimizing Plant Unit 1 completed in 67.3s (Final gap: 0.050%)
```

## 技术实现

### Gap监控原理

1. 创建求解器日志文件
2. 启动后台线程实时读取日志文件
3. 解析求解器输出中的gap信息
4. 使用对数刻度计算进度百分比

### 进度计算公式

```python
progress = (log(initial_gap) - log(current_gap)) / (log(initial_gap) - log(target_gap)) * 100
```

这种方法考虑了MIP gap通常以指数方式减少的特性。

## 配置参数

### Gurobi默认设置

- 目标gap: 0.05% (0.0005)
- 日志文件: 自动创建在logs目录

### CPLEX默认设置  

- 目标gap: 1% (0.01)
- 时间限制: 600秒
- 日志文件: 自动创建在logs目录

## 使用示例

进度监控功能已集成到以下脚本中：

- `main.py`: 单一ISO区域优化
- `cs1.py`: 批量核电站优化
- `cs1_tea.py`: TEA经济分析（简单进度指示）

无需额外配置，运行脚本时会自动显示进度。

## 故障排除

### 问题1: 进度停留在0%

- **原因**: 求解器可能还未开始输出gap信息
- **解决**: 等待求解器初始化完成，通常几秒后会开始显示进度

### 问题2: Gap信息显示为N/A

- **原因**: 无法解析求解器输出或日志文件未创建
- **解决**: 检查求解器版本兼容性，确保有日志文件写入权限

### 问题3: 进度条不显示

- **原因**: tqdm未安装或导入失败
- **解决**: 安装tqdm或使用备用文字显示

### 问题4: 只有第一个反应堆显示进度，后续没有

- **原因**: 日志文件监控线程可能没有正确重启
- **调试**: 查看控制台输出中的调试信息：

  ```
  Starting gap monitoring for gurobi with log file: /path/to/log
  Gap monitoring started for /path/to/log
  Gap found: 15.234% from NEW line: H  150     0...
  ```

- **解决**:
  1. 检查日志目录是否有写入权限
  2. 确认求解器正确创建日志文件
  3. 查看是否有"Gap monitoring finished"消息

### 问题5: 系统读取旧的log信息，gap值不变

- **表现**: Gap值始终显示相同数字，如一直显示"17.400%"
- **原因**: 系统在读取之前优化留下的旧日志内容
- **解决**:
  1. ✅ 已修复：每次优化开始前自动清空日志文件
  2. ✅ 已修复：监控从文件末尾开始，只读取新写入的内容
  3. ✅ 已修复：调试输出显示"NEW line"确认读取的是新内容

### 问题6: 日志文件未创建警告

- **表现**: 看到 "Warning: Log file not created after 30s"
- **原因**: 求解器未启用日志输出或路径错误
- **解决**:
  1. 检查求解器许可证是否正确
  2. 验证日志目录权限
  3. 确认求解器版本支持LogFile参数

## 调试输出说明

运行时会看到以下调试信息：

### 正常情况

```
Starting gap monitoring for gurobi with log file: ../output/cs1/logs/Plant_1_PJM_25_solver_output.log
Gap monitoring started for ../output/cs1/logs/Plant_1_PJM_25_solver_output.log
Starting to monitor from position 0
Gap found: 15.234% from NEW line: H  150     0                    2.234056e+08...
Gap found: 12.456% from NEW line:    200     0 2.100000e+08...
Gap found: 8.123% from NEW line:    250     0 1.950000e+08...
Optimizing Plant Unit 1 completed in 45.2s (Final gap: 0.050%)
Gap monitoring finished. Total NEW gaps found: 28
Optimization completed successfully
```

### 修复后的改进

- **自动清理**: 每次优化自动删除旧日志文件
- **精确监控**: 只读取求解器新写入的内容
- **NEW标记**: 调试输出明确标记读取的是新内容
- **单一进度条**: 避免重复的进度显示，保持输出整洁

## 高级配置

### 自定义目标Gap

```python
# 在求解器选项中修改
solver_options = {"MIPGap": 0.001}  # 0.1% gap target
```

### 调整监控频率

修改代码中的时间间隔：

- 文件检查频率: `time.sleep(0.2)`
- 进度更新频率: `time.sleep(0.5)`

### 启用详细调试

临时启用更多调试输出，在 `_monitor_solver_output` 中将 `gap_found_count <= 3` 改为更大值。

## 性能影响

进度监控功能设计为轻量级：

- 后台线程每0.1秒检查一次日志文件
- 进度条每0.5秒更新一次显示
- 对求解器性能影响微乎其微

## 扩展性

该进度监控系统可以轻松扩展支持其他求解器，只需：

1. 添加对应的日志格式解析函数
2. 在solver_options中配置相应的日志输出参数
3. 更新start()方法中的求解器名称判断逻辑
