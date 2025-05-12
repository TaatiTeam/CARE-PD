# CARE-PD
CARE-PD is a benchmark dataset and evaluation suite for clinical gait analysis in Parkinson’s Disease, released as part of NeurIPS 2025 Datasets & Benchmarks Track submission.

## ⚙️ Get You Ready

<details>
  


```
git clone https://github.com/TaatiTeam/CARE-PD.git
cd CARE-PD
```
### 1️⃣ Install Dependencies

<!-- #### 🔹 Option 1: Install Using Conda (Recommended)
```
conda env create -n archgait -f environment.yml
conda activate archgait
``` -->

We tested our code on Python 3.9.21 and PyTorch 2.6.0

#### 🔹 Install Using Pip
```
python -m venv carepd
source carepd/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```


### 2️⃣ Datasets setup
```
mkdir -p assets/datasets
```
Download the CARE-PD datasets from Dataverse and put them ok in the `assets/datasets` folder.

### 3️⃣ Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh # ToDo
```
</details>

## 🚀 Running code
</details>
