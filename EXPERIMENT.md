# YOLOv8 and YOLOv10 road-damage experiment

This project uses RDD2022, whose Pascal VOC annotations contain four classes:
`D00`, `D10`, `D20`, and `D40`.

## 1. Download data

Download one country archive from the official CRDDC/RDD2022 page:

https://crddc2022.sekilab.global/

Extract it into a temporary folder. Do not commit the dataset to Git.

## 1a. Verify the source data

Before conversion, run the integrity check. It saves class statistics, box
statistics, errors, and 12 annotated control images.

```powershell
python scripts\verify_rdd2022.py `
  --source .\data\raw `
  --output .\reports\verification `
  --samples 12 `
  --seed 42
```

The report is `reports/verification/dataset_verification_report.md` and the
control images are in `reports/verification/samples`.

## 2. Prepare the dataset

The converter creates a deterministic 80/20 train/validation split and converts
the XML bounding boxes to YOLO TXT files.

```powershell
python scripts\prepare_rdd2022.py `
  --source .\data\raw `
  --output .\data\processed\rdd2022 `
  --seed 42
```

For a smoke test, use a small subset first:

```powershell
python scripts\prepare_rdd2022.py `
  --source .\data\raw `
  --output .\data\processed\rdd2022_smoke `
  --max-images 100
```

## 3. Install dependencies

Run these commands from the activated project virtual environment:

```powershell
python -m pip install -r requirements.txt
```

Ultralytics downloads the pretrained `yolov8n.pt` and `yolov10n.pt` files on
the first benchmark run. Internet access is required for that first run.

## 4. Run the comparison

Use a GPU when available:

```powershell
python scripts\benchmark_models.py `
  --data .\data\processed\rdd2022\data.yaml `
  --epochs 30 `
  --batch 8 `
  --device 0
```

For a CPU smoke test:

```powershell
python scripts\benchmark_models.py `
  --data .\data\processed\rdd2022_smoke\data.yaml `
  --epochs 1 `
  --batch 2 `
  --device cpu `
  --output .\runs\smoke
```

The final comparison is saved as `runs/comparison/comparison.json`. Training
configuration is saved as `runs/comparison/experiment_config.json`. Training
plots, `results.csv`, and confusion matrices are saved in the corresponding
model run folders.

## 5. Report the result

Use `precision`, `recall`, `map50`, `map50_95`, validation speed, and model
size in the comparison table. State the exact country subset, random seed,
epochs, image size, batch size, device, and Ultralytics version. Do not compare
the measured values directly with COCO results from the YOLOv10 paper: those
are different datasets and experimental conditions.