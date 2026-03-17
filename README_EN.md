# DG-Mesh Dynamic Gaussian / Mesh Player and Export Tools

[中文说明](./README.md)

---

## 1. Project Scope

This repository is currently a **lightweight utility repo built around DG-Mesh**, mainly containing:

- an **offline exporter** for per-frame Gaussian / Mesh assets
- a **local desktop player** for online Gaussian rendering and offline sequence playback

It is **not a standalone training repository**. The core training, model, and rendering pipeline still come from the original `DG-Mesh` project.

---

## 2. Relationship to DG-Mesh

The current code directly depends on DG-Mesh modules such as:

- `scene`
- `gaussian_renderer`
- `utils.renderer`
- `arguments`

So the safest setup is:

1. keep the original `DG-Mesh` environment available
2. place these scripts into the `dgmesh/` directory of DG-Mesh, or adjust imports to match your own layout

Recommended integrated layout:

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

## 3. Current Features

### Player

- Online Gaussian rendering
- Offline Gaussian sequence playback
- Offline mesh sequence playback
- `Gaussian / Mesh / Split` display modes
- Frame slider, loop playback, FPS control
- Orbit camera interaction
- Basic runtime logging via `--enable-log`

### Exporter

- Export per-frame dynamic Gaussian PLYs
- Export per-frame dynamic mesh PLYs
- Support `uniform / test` time sampling
- Support export stride
- Support preview image export

---

## 4. Runtime Requirements

It is best to reuse the original DG-Mesh environment. The current scripts depend on at least:

- Python
- PyTorch
- CUDA
- PySide6
- trimesh
- plyfile
- imageio
- tqdm
- the original DG-Mesh model and renderer code

> The current rendering path is effectively CUDA-dependent; without CUDA the player may still open, but rendering support is limited.

---

## 5. Launching the Player

### Recommended: run after integrating into DG-Mesh

Run from the DG-Mesh repository root:

```bash
python dgmesh/player.py
```

Optional:

```bash
python dgmesh/player.py --start-dir output --enable-log
```

Arguments:

- `--start-dir`: initial directory for file/folder pickers
- `--enable-log`: write runtime logs to `logs/4dgsplayer.log`

---

## 6. Online Mode

When the player starts:

- the default mode is `Gaussian`
- `Online GS` is enabled by default

Click:

- `Open Online Model Dir`

Then choose the **training output directory itself**. It must contain at least:

```text
cfg_args.txt
point_cloud/
deform/
```

If the directory is valid, the player renders directly in online Gaussian mode.

---

## 7. Offline Export

Recommended command:

```bash
python dgmesh/export_dynamic_assets.py --start_checkpoint output/your_run
```

Common examples:

### Export both Gaussian and Mesh assets

```bash
python dgmesh/export_dynamic_assets.py \
  --start_checkpoint output/your_run \
```

### Export mesh only

```bash
python dgmesh/export_dynamic_assets.py --start_checkpoint output/your_run --export_mesh
```

### Export every 2 frames and save a few preview renders

```bash
python dgmesh/export_dynamic_assets.py \
  --start_checkpoint output/your_run \
  --export_gaussians \
  --export_mesh \
  --export_stride 2 \
  --preview_count 5
```

By default, outputs are written into a new timestamped folder under the checkpoint directory, for example:

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

## 8. Offline Playback

### Gaussian sequence

Click:

- `Open GS Dir`

and choose a directory containing:

```text
frame_0000.ply
frame_0001.ply
frame_0002.ply
...
```

### Mesh sequence

Click:

- `Open Mesh Dir`

and choose a directory using the same `frame_XXXX.ply` naming convention.

### Split mode

If both are available:

- online or offline Gaussian data
- offline mesh data

you can switch to:

- `Split`

for side-by-side playback.

---

## 9. Interaction

- Left mouse drag: orbit
- Right mouse drag: pan
- Mouse wheel: zoom
- `Play`: play / pause
- `Frame`: manual frame control
- `FPS`: playback speed
- `GS Scale`: online Gaussian render resolution scale

---

And thanks to the authors of [DG-mesh](https://github.com/Isabella98Liu/DG-Mesh) and their code!