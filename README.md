# 🔬 Road Damage Detection Benchmark: YOLOv8 vs. YOLOv10

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8_%7C_YOLOv10-00599C?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/Dataset-RDD2022-FF9900?style=for-the-badge)](https://github.com/sekilab/RoadDamageDetector)

---

## 📌 Overview

This research project presents a rigorous comparative evaluation between **YOLOv8** and the end-to-end NMS-free **YOLOv10** architectures for automated road surface defect detection on the **RDD2022 (Road Damage Dataset 2022)**. 

The primary goal is to analyze detection accuracy (`mAP`), model complexity, and inference latency (`ms/frame`) to determine optimal configurations for edge computing and intelligent transportation systems (ITS).

---

## 🗂 Repository Layout

```text
.
├── data/
│   └── raw/               # Local RDD2022 subsets (intentionally ignored by Git)
├── scripts/               # Core Python automation scripts
├── reports/               # Verification reports and analytical drafts
├── EXPERIMENT.md          # Comprehensive step-by-step experiment procedure
└── requirements.txt       # Project dependencies
```

> **⚠️ Version Control Policy:** Raw datasets, trained model weights (`.pt`), virtual environments (`venv/`), and training artifact directories are intentionally excluded from Git. Keep only source code, configs, and analytical reports in the repository.

---

## ⚙️ Quick Start & Execution

Ensure you execute all commands directly from the **project root directory**.

### 1. Environment Setup (Google Colab / Local Linux)
```bash
git clone <PRIVATE_REPOSITORY_URL>
cd <REPOSITORY_NAME>
pip install -r requirements.txt
```

### 2. Pipeline Execution
Place your required RDD2022 subset under `data/raw/`, then run the automation pipeline sequentially:

```bash
# Step 1: Verify data integrity, structure, and label distributions
python scripts/verify_rdd2022.py --source ./data/raw --output ./reports/verification

# Step 2: Preprocess, split, and generate YOLO-compatible structure with a fixed seed
python scripts/prepare_rdd2022.py --source ./data/raw --output ./data/processed/rdd2022 --seed 42

# Step 3: Launch model benchmarking (YOLOv8 vs YOLOv10 on GPU 0)
python scripts/benchmark_models.py --data ./data/processed/rdd2022/data.yaml --device 0
```

---

## 📋 Experiment Documentation

For detailed guidelines regarding hyperparameter settings, loss functions, evaluation metrics (`mAP@50`, `mAP@50-95`), and hardware constraints, please refer to [EXPERIMENT.md](./EXPERIMENT.md).

---

## 👤 Author

**Artur Malakhov**  
B.Sc. Student in Computer Science & Intelligent Control Systems (09.03.01)  
*Institute of Artificial Intelligence, RTU MIREA (Moscow, Russia)*  
🔗 GitHub: [@Chiaarthur1](https://github.com/Chiaarthur1)
