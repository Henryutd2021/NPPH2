# LCOH差异分析与辅助服务统计改进总结

## 1. LCOH差异原因分析

### 问题：为什么Total System的LCOH比Incremental的LCOH更高？

通过代码分析，发现LCOH计算的关键差异在于**成本包含范围**的不同：

#### Total System LCOH计算逻辑（tea.py:1188-1213）
```python
# LCOH calculation - pure cost method, no consideration of revenue and subsidies
if annual_h2_prod_kg > 0:
    # 1. Get all negative cash flows (all costs)
    total_costs = [abs(cf) for cf in cf_array if cf < 0]
    
    # 2. Calculate present value
    pv_total_costs = sum(
        cost / ((1 + discount_rt) ** i) for i, cost in enumerate(total_costs)
    )
```

**包含成本：**
- 所有CAPEX组件（电解器、氢气储存、电池、电网集成、核电站改造）
- 所有运营成本（包括核电站基础运营成本）
- 所有更换和维护成本
- 反映**完整的核氢一体化系统成本**

#### Incremental LCOH计算逻辑（tea.py:1410-1422）
```python
if h2_prod_annual > 0:  # LCOH for incremental H2 project
    # Costs for LCOH are the negative cash flows of the *pure_incremental_cf*
    pv_inc_costs_for_lcoh = sum(
        abs(cf) / ((1 + discount_rt) ** i)
        for i, cf in enumerate(pure_incremental_cf)
        if cf < 0
    )
```

**包含成本：**
- 仅增量CAPEX（电解器、氢气储存、电池系统）
- 仅增量运营成本（VOM、水成本、机会成本）
- 不包括核电站基础设施成本
- 反映**向现有核电站添加氢气生产的边际成本**

### 经济学解释

1. **Total System LCOH**: 新建核氢一体化设施的成本
2. **Incremental LCOH**: 现有核电站改造为氢气生产的成本
3. **差值**: 反映了核电站基础设施成本的分摊

### 现金流计算差异

在`calculate_cash_flows`函数中，Total System包含所有CAPEX组件：
- Electrolyzer_System
- H2_Storage_System  
- Battery_System_Energy & Battery_System_Power
- Grid_Integration
- NPP_Modifications

而在`calculate_incremental_metrics`中，只包含与氢气/电池相关的增量组件。

## 2. 辅助服务统计数据改进

### 新增的AS统计指标

在`calculate_annual_metrics`函数中添加了以下AS相关统计：

#### 收入统计
- `AS_Revenue_Total`: 总AS收入
- `AS_Revenue_Average_Hourly`: 平均小时AS收入
- `AS_Revenue_Maximum_Hourly`: 最大小时AS收入
- `AS_Revenue_Hours_Positive`: AS收入为正的小时数
- `AS_Revenue_Utilization_Rate`: AS收入利用率

#### 单位容量收入
- `AS_Revenue_per_MW_Electrolyzer`: 每MW电解器的AS收入
- `AS_Revenue_per_MW_Battery`: 每MW电池功率的AS收入

#### 投标统计
- `AS_Total_Bid_Services`: 投标的AS服务数量
- `AS_Total_Max_Bid_Capacity_MW`: 总最大投标容量
- `AS_Max_Bid_{service}_MW`: 各服务的最大投标
- `AS_Avg_Bid_{service}_MW`: 各服务的平均投标
- `AS_Bid_Utilization_vs_Electrolyzer`: 投标容量vs电解器容量

#### 部署统计（调度仿真模式）
- `AS_Total_Deployed_Energy_MWh`: 总部署能量
- `AS_Total_Deployed_{service}_MWh`: 各服务的总部署量
- `AS_Avg_Deployed_{service}_MW`: 各服务的平均部署功率
- `AS_Deployment_Efficiency_{service}_percent`: 部署效率（部署/投标）

### 报告改进

在`generate_report`函数中新增了"5. Ancillary Services Performance"部分，包括：

1. **AS收入概览**
   - 总AS收入
   - 平均/最大小时收入
   - 利用率统计

2. **收入构成分析**
   - 各收入源占比
   - AS收入占总收入百分比

3. **投标表现**
   - 投标服务数量
   - 总投标容量
   - 各服务详细投标数据

4. **部署表现**（如适用）
   - 总部署能量
   - 各服务部署详情
   - 部署效率分析

5. **系统利用率**
   - 影响AS能力的关键指标

### 新增LCOH对比分析部分

添加了专门的LCOH分析部分，详细解释：
- Total vs Incremental LCOH数值对比
- 计算方法差异
- 经济学解释
- CAPEX分配明细

## 3. 代码改进位置

1. **tea.py:760-840**: 新增AS统计计算
2. **tea.py:2020-2140**: 改进AS报告部分  
3. **tea.py:2160-2220**: 新增LCOH对比分析
4. **tea.py:2000-2020**: 在性能指标中排除AS收入（避免重复）

这些改进使TEA报告更加全面，特别是对辅助服务的分析更加详细和有用。 