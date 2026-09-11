# Road damage YOLO experiment

Research project comparing YOLOv8 and YOLOv10 for road-damage detection on
RDD2022.

## Repository layout

```text
data/raw/   local RDD2022 subsets; ignored by Git
scripts/    all Python code
reports/    verification report and report drafts
```

The raw dataset, model weights, virtual environment, and generated training
runs are intentionally excluded from Git. Keep only source code and factual
reports in the private repository.

## Run from the project root

```powershell
python scripts/verify_rdd2022.py --source .\data\raw --output .\reports\verification
python scripts/prepare_rdd2022.py --source .\data\raw --output .\data\processed\rdd2022 --seed 42
python scripts/benchmark_models.py --data .\data\processed\rdd2022\data.yaml --device 0
```

See [EXPERIMENT.md](EXPERIMENT.md) for the full experiment procedure.

## Colab workflow

```bash
git clone <PRIVATE_REPOSITORY_URL>
cd pussyproject
pip install -r requirements.txt
```

Then place only the required RDD2022 subset under `data/raw/` and run the
scripts from the project root.