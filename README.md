**机器学习课程作业**

## 1. 如何安装环境  
请使用 **Python 3.9.5**，安装依赖：
```bash
pip install -r requirements.txt
```
## 2. 如何修改参数
编辑 `vessel_analysis.py` 文件，根据需求调整以下参数：
- **数据路径**：
  - 渔船轨迹：
    ```python
    data_path = './data/crop/DCAIS_[30, 1001, 1002]_region_[37.6, 39, -122.9, -122.2]_01-04_to_30-06_trips.csv'
    ```
  - 油轮轨迹：
    ```python
    data_path = './data/crop/DCAIS_[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1017, 1024]_region_[47.5, 49.3, -125.5, -122.5]_01-04_to_30-06_trips.csv'
    ```
- **距离度量方法（四选一）**：
  ```python
  metric = 'dtw'  # 可选值：'dtw'、'dfd'、'hd'、'md'
  ```
- **最小聚类大小（根据船型选择）**：
```python
  msc = 2  # 渔船用2，油轮用3
```

## 3. 如何运行项目

运行以下命令即可启动完整流程：

```bash
python vessel_analysis.py
```

## 4. 项目结构

```text
├── vessel_analysis.py                     # 主程序，执行数据加载、压缩、聚类与可视化
├── requirements.txt                       # 项目依赖文件
├── preprocessing/
│   └── compress_trajectories.py           # 数据预处理与压缩脚本
└── src/
    ├── analysis.py                        # 可视化与性能分析函数
    ├── clustering.py                      # 聚类方法实现（HDBSCAN）
    ├── compression.py                     # 三种压缩算法实现
    └── distance.py                        # 四种轨迹距离度量方法
```
