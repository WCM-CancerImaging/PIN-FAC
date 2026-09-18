# PIN-FAC Inference

Inference-only release of PIN-FAC (Physics-constrained Inference Network for Fatty Acid Composition) for voxel-wise breast adipose fatty acid composition (FAC) mapping from complex multi-echo MRI.

This repository does **not** include training code. A pretrained model checkpoint is applied directly to a single spatial slice represented by separate magnitude and phase DICOM folders.

## Repository structure

```text
PIN-FAC-Inference/
├── infer.py
├── pin_fac_model.py
├── pinfac/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── model_loader.py
│   ├── postprocess.py
│   └── visualize.py
├── weights/
│   └── model.pth              # add pretrained checkpoint here
├── sample/
│   ├── Image/
│   │   ├── *.dcm
│   │   └── ...
│   └── Phase/
│       ├── *.dcm
│       └── ...
├── outputs/
├── requirements.txt
└── README.md
```

## Sample input

The sample represents **one spatial slice** acquired at multiple echo times.

```text
sample/
├── Image/
│   ├── magnitude_echo_01.dcm
│   ├── magnitude_echo_02.dcm
│   └── ...
└── Phase/
    ├── phase_echo_01.dcm
    ├── phase_echo_02.dcm
    └── ...
```

The default model expects **16 echoes** in each folder.

Magnitude files are ordered by the DICOM `EchoTime` field. Phase files also use `EchoTime` when available. For phase exports without `EchoTime`, the code falls back to `EchoNumbers`, `InstanceNumber`, and finally natural filename order. When phase `EchoTime` is present, magnitude/phase echo times are explicitly checked for agreement.

Each DICOM file must contain one 2-D image for one echo of the same slice.

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Place the pretrained model checkpoint at:

```text
weights/model.pth
```

## Inference

From the repository root:

```bash
python infer.py \
    --image-dir sample/Image \
    --phase-dir sample/Phase \
    --weights weights/model.pth \
    --out-dir outputs/sample
```

GPU inference is used automatically when CUDA is available. To force CPU inference:

```bash
python infer.py \
    --image-dir sample/Image \
    --phase-dir sample/Phase \
    --weights weights/model.pth \
    --out-dir outputs/sample \
    --device cpu
```

## Outputs

The default output directory contains:

```text
outputs/sample/
├── FF.npy
├── SFA.npy
├── MUFA.npy
├── PUFA.npy
├── R2.npy
├── ndbb.npy
├── nmidbb.npy
├── mask2d.npy
├── te_ms.npy
├── preview.png
└── manifest.json
```

`preview.png` uses the following layout:

```text
Echo-1 | FF   | R2*  | Ndb
Nmidb  | SFA  | MUFA | PUFA
```

Use `--save-all` to additionally save water, fat, UFA, frequency, phase-related outputs, and other derived maps.

## Preprocessing

For each voxel, the complex multi-echo input is constructed as

```text
magnitude_normalized(e) = magnitude(e) / magnitude(echo 1)

x(e) = [
    magnitude_normalized(e) * cos(phase(e)),
    magnitude_normalized(e) * sin(phase(e))
]
```

The anchor input is

```text
anchor = log(magnitude(echo 1))
```

The default inference mask follows the preprocessing used for model development:

```text
magnitude(echo 1) > 50
```

and

```text
magnitude(echo 2) > 0.08 × magnitude(echo 1)
```

These values can be changed with `--m0-threshold`, `--mask-echo-idx`, and `--mask-rel-to-m0`.

## FAC conversion

The model predicts nine normalized parameters. After output scaling, FAC is calculated from `Ndb` and `Nmidb` as

```text
UFA  = (Ndb - Nmidb) / 3
PUFA = Nmidb / 3
SFA  = 1 - UFA
MUFA = UFA - PUFA
```

The output scaling constants in `pinfac/postprocess.py` are the same values used during model training.

## Important input assumptions

- The magnitude and phase folders correspond to the **same slice**.
- Each folder contains one DICOM file per echo.
- Magnitude and phase series contain the same echo times.
- The default pretrained model expects 16 echoes.
- Phase data must be either in radians or in the Siemens-like 12-bit range used by the preprocessing pipeline.

If the acquisition or DICOM export format differs from these assumptions, the preprocessing code should be adapted before applying the pretrained model.

## Citation

If you use this code, please cite the associated paper once publication information is available.
