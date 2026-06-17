
# SlowSight


[![FAST'27 Artifact](https://img.shields.io/badge/FAST-2027-blue.svg)]()
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)]()

---

# Artifact Overview

This artifact accompanies the FAST'27 paper:

> **Labeling the Invisible: A Scalable Framework for Labeling Fail-Slow Failures in Cloud Storage Systems**

SlowSight is a semi-automated fail-slow labeling framework designed to assist operators in efficiently and consistently identifying fail-slow failures in large-scale cloud storage systems. Rather than labeling individual anomalies one by one, SlowSight combines automated anomaly candidate extraction, domain knowledge, and pattern-centric labeling to reduce manual effort while preserving labeling quality.

Consistent with the paper, SlowSight consists of these major stages:

1. **Data Preparation**
2. **Anomaly Candidate Extraction**
   - **Historical Deviation Detection**
   - **Peer Component Comparison**
3. **Pattern-Centric Labeling**
   - **Knowledge-Driven Filtering**
   - **Anomaly Segment Construction and Representation**
   - **Scalable Sampling and Labeling**
   - **Interactive Labeling Interface**

The artifact includes the implementation of the framework, the interactive labeling system, and the experimental environment dataset (D3) used for evaluation.

---

## 1. Getting Started Instructions (≈ 30 minutes)

This section guides users through the minimal steps to verify the artifact runs,
produces meaningful output, and leaves no obvious blockers.

### 1.1 Prerequisites

* **OS**: Linux 
* **Conda / Miniconda 3** (recommended) or a system-wide Python 3.8 installation.
* **git** (for cloning).

### 1.2 Step 1 — Download the repository

Please download the archive of this repository, extract its contents, and then change the working directory to the extracted folder.Alternatively, you may also utilize Git to clone the remote repository.
```bash
git clone https://github.com/AIOps-Lab-NKU/SlowSight.git
cd SlowSight
```

### 1.3 Step 2 — Set up the Python environment (≈ 15 minutes)

```bash
cd backEnd
conda create -n slowSight python=3.8 -y
conda activate slowSight
pip install -r requirements.txt
# If the download fails or proceeds at a low speed, you may switch to a domestic mirror source for installation.
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> If `conda install patool` reports "package not found", use `pip install patool` as
> a fallback. The package is only required for archive extraction and is non-critical.

### 1.4 Step 3 — Verify the sample data

The D3 dataset is shipped under `backEnd/data/`:
```
backEnd/data/
├── D3_metric.csv              # Main time-series file 
├── D3_metric_candidates.json  # Column metadata & candidate metric list
└── ground_truth.csv           # Labeled ground-truth anomaly segments
```
Expected: a CSV whose first column is `timestamp` and remaining columns are per-host

### 1.5 Step 4 — Launch the interactive visualization (≈ 5 minutes)

As this project follows a frontend‑backend separation architecture, two terminals are required to run simultaneously — one for the frontend and one for the backend—to ensure proper collaborative operation.
```bash
# Terminal 1: backend
cd backEnd
set FLASK_APP=app.py
flask run

#Terminal 2: frontend
cd frontEnd
# If npm is not available in the current environment, it must be installed beforehand.(Optional)
conda install -c conda-forge nodejs -y
npm install
npm run dev
```

### 1.6 Step 5 — Use SlowSight to label datas (≈ 15 minutes)

Please navigate to the “Line Chart Label” tool, select “D3” from the “Data Selection” dropdown menu and “SlowSight” from the “Algorithm Selection” dropdown menu, and finally click the “label” button to perform the labeling operation.

To expedite the process, you may use the pre‑generated model directly. If regeneration of model is required, please allow an additional 20 minutes. The code required for this operation is located in the run function within the main.py file, specifically in the commented‑out section labeled “Train the model”.

---
### 1.7 Demo Video

Our demo video is located in the assets folder and is available for you to download and view.

---

# 2. Framework Design

This section briefly introduces the design of SlowSight following the structure of Section 4 in the paper.

## 2.1 Data Preparation

Reliable monitoring data is the foundation of fail-slow labeling.

In production environments, monitoring streams frequently contain missing values, invalid observations, and scale inconsistencies. SlowSight therefore performs a preprocessing stage before anomaly analysis.

### Processing Steps

1. Filter invalid values.
2. Linear interpolation for short missing intervals.
3. Min-max normalization.

The resulting data provide consistent inputs for subsequent detection and labeling modules.

---

## 2.2 Anomaly Candidate Extraction

Fail-slow failures exhibit diverse manifestations. To capture these behaviors, SlowSight adopts a **multi-view anomaly candidate extraction strategy** that combines:

- Historical Deviation Detection
- Peer Component Comparison

The two views complement each other and improve robustness under heterogeneous workloads.

### Historical Deviation Detection

This module detects deviations from a component's own historical behavior.

Key components include:

#### Variate Attention-Based Encoder

Instead of modeling temporal dependencies alone, SlowSight models inter-metric relationships by treating metrics as variates and explicitly learning their correlations.

#### Adversarial Dual-Decoder

The reconstruction framework contains:

- Encoder
- Generator Decoder
- Discriminator Decoder

Adversarial training improves sensitivity to gradual fail-slow degradation.

#### Anomaly Score Calculation and Threshold Selection

SlowSight computes reconstruction-based anomaly scores and employs:

- **SPOT (Streaming Peaks-Over-Threshold)**

to derive adaptive thresholds suitable for dynamic cloud environments.

#### Transfer Learning

To improve scalability, pretrained models are transferred across homogeneous component types, reducing training costs in large deployments.

---

### Peer Component Comparison

Historical behavior is not always available or reliable.

To complement temporal modeling, SlowSight compares a component against its peers operating under similar conditions.

#### Clustering-Based Outlier Identification

The workflow includes:

1. Statistical feature extraction
2. PCA dimensionality reduction
3. DBSCAN clustering

Potential anomalous components are identified as behavioral outliers.

#### Adaptive Outlier Filtering

Three stages are applied:

1. Cluster Categorization
2. Distance-Based Outlier Determination
3. Temporal Consistency Validation

This mechanism reduces transient false positives while preserving fail-slow candidates.

---

## 2.3 Pattern-Centric Labeling

Detecting anomalies alone is insufficient because many anomalies are unrelated to fail-slow failures.

SlowSight therefore reformulates labeling as a **representative anomaly pattern identification problem**.

### Knowledge-Driven Filtering

Statistical anomalies are first examined from the perspective of fail-slow semantics.

SlowSight:

- Extracts trend features using TSFRESH.
- Applies meaning-aware filtering rules.
- Removes anomaly candidates inconsistent with fail-slow behavior.

Only semantically meaningful candidates proceed to the next stage.

---

### Anomaly Segment Construction and Representation

For each retained candidate:

1. An anomaly onset point is identified.
2. A fixed-length multi-metric window is extracted.
3. TSFRESH features are computed.
4. A unified anomaly segment representation is generated.

This representation enables robust comparison across components and workloads.

---

### Scalable Sampling and Labeling

SlowSight clusters anomaly segments into representative patterns.

The implementation follows the paper and uses:

- Hierarchical Agglomerative Clustering (HAC)
- Davies–Bouldin Index

Operators label only cluster centroids, and labels are propagated to all corresponding members.

This design dramatically reduces manual labeling effort.

---

### Interactive Labeling Interface

SlowSight provides an interactive graphical interface that enables operators to:

- Inspect anomaly clusters
- Compare components
- Visualize multi-metric anomaly segments
- Validate representative patterns

The interface supports efficient human-in-the-loop labeling in production environments.

---

# 3. Project Structure

```text
SlowSight/
├── backEnd/
│   ├── SlowSight/
│   ├── data/
│   ├── model/
│   ├── result/
│   ├── app.py
│   └── requirements.txt
│
├── frontEnd/
│
└── README.md
```

---

# 4. End-to-End Workflow

The complete workflow follows the architecture presented in in this figure.

<img src="./assets/framework.png" width="750">

1. Data Preparation
2. Historical Deviation Detection
3. Peer Component Comparison
4. Knowledge-Driven Filtering
5. Anomaly Segment Construction and Representation
6. Scalable Sampling and Labeling

The final output is a set of fail-slow labels generated through pattern-centric labeling and operator verification.

---
# 5. Datasets
The **D3** dataset is included in this repository, which can be used to validate SlowSight accordingly.

## D3: Failure Injection Dataset

Constructed using NIC fail-slow fault injection.

The dataset encompasses the following seven metrics across 11 nodes: "PassiveOpens", "orphan", "TCPSynRetrans", "TCPTimeouts", "InSegs", "RetransSegs", and "OutSegs".

Used to evaluate the generalizability of SlowSight.

---
# 6. Artifact Claims

This artifact supports the following claims from the paper.

## Claim 1: Effective Fail-Slow Labeling

SlowSight achieves strong fail-slow labeling performance on both production and public datasets.

## Claim 2: Contribution of Each Module

Historical Deviation Detection, Peer Component Comparison, and Knowledge-Driven Filtering all contribute to overall performance.

## Claim 3: Reduced Labeling Effort

Pattern-Centric Labeling significantly reduces manual inspection workload while maintaining label quality.

## Claim 4: Generalizability

SlowSight generalizes beyond storage devices and remains effective for NIC-related fail-slow failures.

---
## 7. License & Acknowledgements

The SlowSight artifact is released under the MIT License. 

This repository is constructed based on the following repos:

- Time-Series-Library: [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- SPOT: [SPOT](https://github.com/limjcst/ads-evt)

We thank the authors for open-sourcing their work.

---
