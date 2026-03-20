
# DGmeshPlayer
DG-Mesh Dynamic Gaussian / Mesh Player and Export Tools
=======
# DG-Mesh 动态高斯 / 网格播放器与导出脚本

[English Version](./README_EN.md)

---
tips:新的分支使用了react嵌入qt
---
## 1. 项目定位

这个仓库目前是一个 **围绕 DG-Mesh 的轻量功能仓库**，主要包含两部分：

- **离线导出器**：把 DG-Mesh 的动态结果导出成逐帧 Gaussian / Mesh 资产
- **本地播放器**：用于预览在线 Gaussian 渲染，或播放离线导出的逐帧序列

它**不是独立训练仓库**，核心训练、模型、渲染模块仍然依赖原始 `DG-Mesh` 工程。


---

## 2. 与 DG-Mesh 的关系

当前代码直接依赖 DG-Mesh 内的模块，例如：

- `scene`
- `gaussian_renderer`
- `utils.renderer`
- `arguments`

所以**最稳妥的使用方式**是：

1. 保持原始 `DG-Mesh` 环境可用
2. 将本仓库中的脚本放入 DG-Mesh 的 `dgmesh/` 目录中，或按你自己的目录结构调整导入路径

推荐集成后的结构：

```text
DG-Mesh/
├─ dgmesh/
│  ├─ player.py
│  ├─ player_ui.py
│  ├─ export_dynamic_assets.py
│  └─ ...
└─ output/
```

---

## 3. 当前功能

### 播放器

- Online Gaussian 渲染
- Offline Gaussian 序列播放
- Offline Mesh 序列播放
- `Gaussian / Mesh / Split` 三种显示模式
- 帧滑动、循环播放、FPS 控制
- 轨道相机交互
- 简单日志输出（`--enable-log`）

### 导出器

- 导出逐帧动态 Gaussian PLY
- 导出逐帧动态 Mesh PLY
- 支持 `uniform / test` 两种时间采样方式
- 支持导出步长
- 支持预览图导出

---

## 4. 运行环境

建议直接复用 DG-Mesh 原有环境。当前脚本依赖至少包括：

- Python
- PyTorch
- CUDA
- PySide6
- trimesh
- plyfile
- imageio
- tqdm
- DG-Mesh 原仓库中的模型与渲染代码

> 当前渲染链路基本默认 CUDA 可用；没有 CUDA 时播放器能启动，但渲染能力会受限。

---

## 5. 启动播放器

### 推荐方式：集成到 DG-Mesh 后启动

在 DG-Mesh 仓库根目录运行：

```bash
python dgmesh/player.py
```

可选参数：

```bash
python dgmesh/player.py --start-dir output --enable-log
```

参数说明：

- `--start-dir`：文件选择框初始目录
- `--enable-log`：将运行日志写入 `logs/4dgsplayer.log`

---

## 6. Online 模式使用方法

播放器启动后：

- 默认模式为 `Gaussian`
- 默认勾选 `Online GS`

点击：

- `Open Online Model Dir`

然后选择 **某一次训练输出目录本身**，目录中至少需要包含：

```text
cfg_args.txt
point_cloud/
deform/
```

如果目录合法，播放器会直接执行在线 Gaussian 渲染。

---

## 7. Offline 导出

推荐使用：

```bash
python dgmesh/export_dynamic_assets.py --start_checkpoint output/your_run
```

常见示例：

### 同时导出 Gaussian 与 Mesh

```bash
python dgmesh/export_dynamic_assets.py ^
  --start_checkpoint output/your_run ^
```

### 仅导出 Mesh

```bash
python dgmesh/export_dynamic_assets.py --start_checkpoint output/your_run --export_mesh
```

### 每 2 帧导出一次，并保存少量预览图

```bash
python dgmesh/export_dynamic_assets.py ^
  --start_checkpoint output/your_run ^
  --export_gaussians ^
  --export_mesh ^
  --export_stride 2 ^
  --preview_count 5
```

默认情况下，输出会写到 checkpoint 目录下一个新的时间戳文件夹中，例如：

```text
output/your_run/
└─ dynamic_assets_offline_2000-01-1_12-00-00/
   ├─ dynamic_gaussians_offline/
   │  ├─ frame_0000.ply
   │  ├─ frame_0001.ply
   │  └─ preview_renders/
   └─ dynamic_mesh_offline/
      ├─ frame_0000.ply
      ├─ frame_0001.ply
      └─ preview_renders/
```

---

## 8. Offline 播放

### Gaussian 序列

点击：

- `Open GS Dir`

选择一个包含如下文件的目录：

```text
frame_0000.ply
frame_0001.ply
frame_0002.ply
...
```

### Mesh 序列

点击：

- `Open Mesh Dir`

选择一个同样采用 `frame_XXXX.ply` 命名规则的目录。

### Split 模式

如果同时加载了：

- Online 或 Offline Gaussian
- Offline Mesh

即可切换到：

- `Split`

进行左右对照播放。

---

## 9. 交互说明

- 鼠标左键拖动：轨道旋转
- 鼠标右键拖动：平移
- 滚轮：缩放
- `Play`：播放 / 暂停
- `Frame`：手动切帧
- `FPS`：设置播放速度
- `GS Scale`：调整 online Gaussian 渲染分辨率缩放

---

最后，感谢[DG-mesh](https://github.com/Isabella98Liu/DG-Mesh)的作者及代码

