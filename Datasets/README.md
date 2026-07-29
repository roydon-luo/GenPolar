# Dataset Layout

`Datasets/` is intentionally distributed without data. Training and test sets
are separated first, followed by the dataset name:

```text
Datasets/
|-- train/
|   |-- dataset_a/
|   `-- dataset_b/
`-- test/
    |-- dataset_a/
    `-- dataset_b/
```

The recommended training path is therefore
`Datasets/train/<dataset_name>/<data>`. Point each `dataroot_gt` in
`data_config.yaml` to one dataset directory, such as
`./Datasets/train/dataset_a`; do not point it to the shared `train/` directory.

The loader accepts the following three formats. Do not mix formats within one
scene directory.

## 1. Stokes arrays (`.npy`)

Store one scene per file directly below the configured data root:

```text
Datasets/
`-- train/
    `-- dataset_a/
        |-- scene_0001.npy
        `-- scene_0002.npy
```

Each array must contain RGB `S0`, `S1`, and `S2` in either layout:

- `(H, W, 3, 3)`, where axis 2 is `(S0, S1, S2)` and the last axis is RGB;
- `(H, W, 9)`, with channels ordered as
  `(S0_R, S0_G, S0_B, S1_R, S1_G, S1_B, S2_R, S2_G, S2_B)`.

Floating-point arrays are recommended. `S1` and `S2` may be signed. If the
reconstructed analyzer intensities exceed 1, all three Stokes components are
scaled together to preserve their physical ratios.

## 2. Stokes images

Store one scene per subdirectory using these exact lowercase PNG file names:

```text
Datasets/
`-- train/
    `-- dataset_a/
        |-- scene_0001/
        |   |-- s0.png
        |   |-- s1.png
        |   `-- s2.png
        `-- scene_0002/
            |-- s0.png
            |-- s1.png
            `-- s2.png
```

All three files must be RGB images with identical dimensions. The stored
values are decoded as `S0 = 2 * s0`, `S1 = 2 * s1 - 1`, and
`S2 = 2 * s2 - 1`; therefore `s0.png`, `s1.png`, and `s2.png` use the standard
8-bit range `[0, 255]`.

## 3. Four analyzer-angle images

Store one scene per subdirectory using these exact PNG file names:

```text
Datasets/
`-- train/
    `-- dataset_a/
        |-- scene_0001/
        |   |-- 0.png
        |   |-- 45.png
        |   |-- 90.png
        |   `-- 135.png
        `-- scene_0002/
            |-- 0.png
            |-- 45.png
            |-- 90.png
            `-- 135.png
```

The four RGB images must have identical dimensions and represent analyzer
angles of 0, 45, 90, and 135 degrees. The loader computes:

```text
S0 = (I0 + I45 + I90 + I135) / 2
S1 = I0 - I90
S2 = I45 - I135
```

For every format, each image must be at least as large as `gt_size` in both
dimensions because training uses an aligned random crop. The public
configuration sets `gt_size: 512`.

To add multiple roots, extend `data_config.yaml`:

```yaml
data_source:
  source1:
    dataroot_gt: ./Datasets/train/dataset_a
  source2:
    dataroot_gt: ./Datasets/train/dataset_b
  source3:
    dataroot_gt: ./Datasets/train/dataset_c
  source4:
    dataroot_gt: ./Datasets/train/dataset_d
```

Use the same `Datasets/test/<dataset_name>/<data>` structure for evaluation
data. The bundled Quick Test remains independent of these directories and
uses the two images under `examples/`.
