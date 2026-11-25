# 断路器双稳态仿真

当前项目将断路器动力学、Isaac 仿真采集和结果分析分开。Isaac 的启动、USD 加载、关节设置和逐物理步采集统一放在 `simulation/isaac_collector.py`。

## 目录

- `core/`：双稳态弹簧、工况参数和单自由度理论求解器，不依赖 Isaac Sim。
- `simulation/`：Isaac 仿真运行与数据采集。
- `analysis/`：对比图和 CSV 导出。
- `analysis/compare.py`：角度、角速度误差统计。
- `configs/default.yaml`：当前基准工况配置。
- `runs/`：每次运行生成独立的结果目录。

## 运行

在 Isaac Lab 的 Python 环境中，从仓库根目录运行：

```bash
python -m project.run
```

运行会读取仓库根目录下的 `breaker.usd`，并把 CSV、对比图和元信息写入 `project/runs/`。

## 后续扩展

未来接入 Isaac Lab visual-tactile 时，将传感器数据及其时间戳加入采集结果，并与拨片状态序列关联。手指训练环境和控制接口等到训练阶段再增加。

