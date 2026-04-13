<!-- HEADER BANNER -->
<div align="center">
<img src="https://capsule-render.vercel.app/api?type=venom&height=200&color=0:0D1117,50:003344,100:00D9FF&text=Adnane%20Erekraken&fontColor=00D9FF&fontSize=42&fontAlignY=55&desc=AI%20Engineer%20%7C%20MSc%20AI%20%40%20CentraleSupélec&descColor=8B9EA8&descSize=16&descAlignY=75&animation=fadeIn&stroke=00D9FF&strokeWidth=1"/>
</div>

<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=15&duration=3500&pause=800&color=00D9FF&center=true&vCenter=true&width=650&lines=Foundation+Models+%7C+Sim-to-Real+Transfer+%7C+Strategic+AI;%24+python+train.py+--mode+pretrain+--dataset+CWRU+CMAPSS+MFPT;%24+git+push+origin+main+%23+another+late+night+commit;Pretrain+once.+Fine-tune+anywhere.+Deploy+everywhere.)](https://git.io/typing-svg)

<br/>

[![LinkedIn](https://img.shields.io/badge/─%20linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adnane-erekraken/)
[![GitHub](https://img.shields.io/badge/─%20github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AdnaneErek)
[![Email](https://img.shields.io/badge/─%20mail-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:adnane.erekraken@student-cs.fr)

</div>

---

## `> whoami`

```python
class AdnaneErekraken:
    def __init__(self):
        self.name       = "Adnane Erekraken"
        self.role       = "AI Engineer"
        self.education  = ["MSc AI @ CentraleSupélec", "General Engineering @ ECC"]
        self.focus      = ["Foundation Models", "Time-Series AI", "Strategic AI Systems"]
        self.languages  = ["Python", "PowerShell", "SQL"]
        self.location   = "Paris, France 🇫🇷"

    def current_work(self):
        return [
            "⚙️  GPT-inspired Foundation Model for industrial PHM (CWRU, CMAPSS, MFPT)",
            "🌟  POLARIS — AI-powered Strategic Steering & Portfolio Optimization Platform",
        ]

    def philosophy(self):
        return "Bridge cutting-edge AI research with real business impact."
```

---

## `> ls -la ./projects`

### `📁 POLARIS/` — Portfolio Optimization & Learning AI for Risk-Adjusted Strategy
> *Enterprise-grade AI platform for strategic decision-making — a North Star for steering committees.*

- **Monte Carlo** option evaluation with **CVaR10** risk-adjusted scoring
- Closed-loop **learning system**: outcome capture → forecast recalibration → scoring adaptation
- Executive-grade **PDF SteerCo reports**, interactive dashboards & conversational interface
- Built-in **guardrails**: budget, capacity, compliance & governance checks
- Stack: `Python` `Pandas` `scikit-learn` `Streamlit` `Plotly` `ReportLab` `Gemini API`

---

### `📁 PHM-FoundationModel/` — Fault Diagnosis & Prognosis @ LGI Lab, CentraleSupélec
> *6-month research project (Oct 2025 – Mar 2026) · Team of 4 · HPC-scale training · Two parallel research tracks*

**The core problem:** Traditional PHM models fail to generalize — every new machine or fault type demands a fresh labeled dataset. This mirrors the pre-GPT era in NLP. The goal: a model that pretrains once and transfers everywhere.

<details>
<summary><code>$ cat phase1_leakage_investigation.md</code></summary>

<br/>

> Near-perfect accuracy (97–99.99%) in PHM literature looked suspicious. We dug in.

**Two systematic leakage sources identified & corrected:**

| Leakage Type | Root Cause | Fix |
|---|---|---|
| **Overlapping window leakage** | Overlapping train/test windows from same signal → model memorizes waveforms | Non-overlapping windows + signal-level split |
| **Bearing-level leakage** | CWRU Drive End & Fan End sensors record same bearing, split independently | Group-wise split by physical bearing identity |

**Quantified impact on MOMENT accuracy (CWRU):**

```
Standard protocol (both leakages)  →  98.97%  ████████████████████
Fix bearing-level only             →  89.70%  ██████████████████
Fix window-level only              →  83.46%  █████████████████
Fully leak-free protocol           →  78.73%  ████████████████
                                              ↑ 20pp gap = artifactual
```

> **Finding:** A large fraction of reported SOTA in PHM deep learning is artifactual. Group-wise splitting by physical unit identity should be the community standard.

</details>

<details>
<summary><code>$ cat phase2_architecture_benchmark.md</code></summary>

<br/>

**Models benchmarked under leak-free protocol:**

| Model | Pretraining | Accuracy (leak-free) |
|---|---|---|
| GPT4TS | Time-series patches on GPT-2 backbone | — |
| **MOMENT** | Masked autoencoder on large TS corpus | **78.73%** → **97.38%** (multi-label) |
| NuTime | Transformer on UCR/UEA datasets | — |
| RoBERTa | Text pretrained, fine-tuned on TS | ~96% (leaky) |

**Task reformulation → multi-label classification:**
Each sample gets binary vector `y = [Ball, InnerRace, OuterRace]` — physically realistic, handles co-occurring faults, reduces class imbalance.
Result: **97.38% subset accuracy, Macro-AUC 0.9996** under leak-free protocol.

**Fine-tuning strategy — Progressive Unfreezing:**
```
Stage 1 → Linear probe      (encoder frozen, head only)
Stage 2 → Partial unfreeze  (last transformer block)
Stage 3 → Full unfreeze     (entire encoder)
```
Preserves pretrained representations. Avoids catastrophic forgetting.

</details>

<details>
<summary><code>$ cat track2_robot_sim_to_real.md</code> &nbsp;← Adnane's track</summary>

<br/>

**Setup:** 4-DOF robot arm · 9 fault classes (healthy + 4 stuck + 4 steady-state error) · 9-channel trajectory signals · Sim data only for training, real robot for eval.

**The gap problem:**
```bash
$ python eval.py --model scratch --test real
# Sim: 81.94%  /  Real: 55.56%  /  Gap: 26.4pp

$ python eval.py --model scratch+gaussian_noise --test real
# Sim: up       /  Real: 48.89%  /  Gap WIDENS  <- generic noise fails
```

**Root cause — what makes real != sim:**
- Actuation delays: sim used fixed 10-step delay → real robot has variable delays
- Healthy motor noise: sim was deterministic → real has Gaussian steady-state errors

**Physics-informed Simulink augmentation:**
```python
holdingTimeIdx  ~ Uniform(2, 30)          # variable actuation delay
normalMotorErr  ~ Gaussian(mu=0, sigma=1.2)  # discretized to [-3,...,3]
faultMagnitude  ~ Uniform(10, 50)         # expanded fault range
```

**Training strategy comparison:**

| Strategy | Sim Acc | Real Acc | Gap |
|---|---|---|---|
| A1 — Scratch + Full FT (baseline) | 81.94% | 57.78% | 24.2pp |
| A2 — Scratch + Head Only | **90.19%** | 61.11% | 29.1pp |
| A3 — Supervised Pretrain + 25% unfreeze | 82.96% | 67.78% | 15.2pp |
| **A4 — Supervised Pretrain + 50% unfreeze** | 83.33% | **70.00%** | **13.3pp** ✅ |

> **Key insight:** A2 scores 90.19% sim but only 61.11% real. Fitting the simulation too tightly is **counterproductive**. Pretraining + partial unfreezing gains **+12pp real accuracy**.

**Self-supervised objectives — both collapsed:**
```bash
$ python train.py --objective next_token_prediction   # ~11% (random chance)
$ python train.py --objective masked_token_prediction  # ~11% (random chance)
# Trajectory fault signatures require labeled pretraining.
# Generic reconstruction losses don't learn fault-discriminative features.
```

</details>

<details>
<summary><code>$ cat track1_multidataset_foundation_model.md</code></summary>

<br/>

**Datasets aggregated for pretraining:**
`CWRU` · `PRONOSTIA/FEMTO` · `CMAPSS (NASA)` · `MFPT` · *(gear & battery in progress)*

**Objective:** Masked reconstruction pretraining across heterogeneous PHM domains — mechanical vibration → thermal degradation → aerodynamic performance.

**Key result:** Multi-dataset pretrained model outperforms single-dataset pretraining on **held-out datasets not seen during pretraining** — early evidence of genuine cross-domain transfer.

**Scale:** 12M+ rows · ~2 hrs/epoch on single GPU · La Ruche HPC (SLURM)

</details>

**Stack:**
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![MATLAB](https://img.shields.io/badge/Simulink-0076A8?style=flat-square&logo=mathworks&logoColor=white)
![Linux](https://img.shields.io/badge/HPC_SLURM-FCC624?style=flat-square&logo=linux&logoColor=black)

---

### `📁 Bach2BeethovenAI/` — AI Music Generation
> *BiLSTM trained on MIDI corpora to generate stylistically consistent classical compositions.*

- Tokenization via `music21`, temperature-based sampling, MIDI → WAV automation
- Stack: `PyTorch` `TensorFlow` `music21`

---

### `📁 Melanoma_Detection_ISIC2019/` — Image + Metadata Fusion with ResNet-50 & Lightning
> *Multimodal deep learning pipeline for the ISIC 2019 melanoma classification challenge.*

Fuses **dermoscopic images** (224×224, hair-removed) with **patient metadata** (age, sex, anatomical site) through a late-fusion classifier trained on multi-GPU nodes with PyTorch Lightning DDP.

<details>
<summary><code>$ cat architecture.md</code></summary>

<br/>

**Multimodal fusion pipeline:**
```
Images (224×224) ──► ResNet-50 (ImageNet pretrained) ──► 2048-dim features ──┐
                                                                               ├──► Fusion (256) ──► 8 classes
Metadata (age/sex/site) ──► MLP encoder ──────────────── 128-dim features ──┘
```

**Two-stage training strategy:**
```
Epoch 0–2  →  Warm-up       (backbone frozen, head only)
Epoch ≥3   →  Fine-tuning   (full model, layer-wise LR decay)
```

**Metrics logged to TensorBoard:** Accuracy · Macro F1 · Weighted F1 · Per-class P/R/F1 · AUROC · Confusion matrix

**Grad-CAM** visualization — side-by-side (original + heatmap) for model interpretability.

**Distributed training:** DDP-compatible with SLURM (`srun python ...`)

</details>

<details>
<summary><code>$ tree Melanoma_Detection_ISIC2019/</code></summary>

<br/>

```
Melanoma_Detection_ISIC2019/
├── src/
│   ├── train/
│   │   ├── train.py              # Training script
│   │   ├── lightning_module.py   # Lightning module
│   │   └── gradcam.py            # Grad-CAM script
│   ├── models/
│   │   └── combined_model.py     # ResNet + MLP fusion
│   └── dataset/
│       └── isic_dataset.py       # Image+metadata dataset
├── processed/
│   ├── images_224_nohair/        # Preprocessed images
│   └── splits/
│       ├── train.csv
│       └── val.csv
├── checkpoints/                  # Best and final models
├── logs/                         # TensorBoard + CSV logs
├── requirements.txt
└── README.md
```

</details>

**Stack:**
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?style=flat-square&logo=lightning&logoColor=white)
![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Linux](https://img.shields.io/badge/HPC_DDP-FCC624?style=flat-square&logo=linux&logoColor=black)

---

## `> cat ./stack.json`

<div align="center">

**`// languages & frameworks`**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**`// data & viz`**

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![PowerBI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

**`// devops & infra`**

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![GitLab](https://img.shields.io/badge/GitLab-FC6D26?style=for-the-badge&logo=gitlab&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![PowerShell](https://img.shields.io/badge/PowerShell-5391FE?style=for-the-badge&logo=powershell&logoColor=white)

**`// platforms`**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge&logo=mathworks&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## `> htop --filter=github`

<div align="center">

<img height="180em" src="https://github-readme-stats.vercel.app/api?username=AdnaneErek&show_icons=true&theme=tokyonight&count_private=true&hide_border=true&title_color=00D9FF&icon_color=00D9FF&rank_icon=github&cache_seconds=1800"/>
<img height="180em" src="https://github-readme-stats.vercel.app/api/top-langs/?username=AdnaneErek&layout=compact&theme=tokyonight&hide_border=true&title_color=00D9FF&cache_seconds=1800"/>

</div>

<div align="center">

[![GitHub Streak](https://streak-stats.demolab.com/?user=AdnaneErek&theme=tokyonight&hide_border=true&ring=00D9FF&fire=00D9FF&currStreakLabel=00D9FF)](https://git.io/streak-stats)

</div>

---

## `> crontab -l  # certifications`

| `cert_id` | Title | Issuer |
|-----------|-------|--------|
| `BCG-DS` | Data Science Job Simulation | Boston Consulting Group |
| `BCG-SC` | Strategy Consulting Job Simulation | Boston Consulting Group |
| `MS-PBI` | Power BI Data Analyst Professional Certificate | Microsoft |
| `GGL-PM` | Google Project Management Specialization | Google |
| `UA-SWE` | Client Needs & Software Requirements | University of Alberta |

---

## `> tail -f /var/log/activity.log`

<div align="center">

[![Adnane's Activity Graph](https://github-readme-activity-graph.vercel.app/graph?username=AdnaneErek&theme=tokyo-night&hide_border=true&area=true&custom_title=──%20commit%20log%20──)](https://github.com/AdnaneErek)

</div>

---

## `> ./snake --watch`

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AdnaneErek/AdnaneErek/output/github-contribution-grid-snake-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/AdnaneErek/AdnaneErek/output/github-contribution-grid-snake.svg" />
  <img alt="github-snake" src="https://raw.githubusercontent.com/AdnaneErek/AdnaneErek/output/github-contribution-grid-snake-dark.svg" />
</picture>

</div>

---

## `> fortune | cowsay`

<div align="center">

[![Readme Quotes](https://quotes-github-readme.vercel.app/api?type=horizontal&theme=tokyonight)](https://github.com/piyushsuthar/github-readme-quotes)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00D9FF,50:003344,100:0D1117&height=120&section=footer&animation=twinkling"/>

![Visitor Count](https://komarev.com/ghpvc/?username=AdnaneErek&color=00d9ff&style=for-the-badge&label=PROFILE+VIEWS)

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&duration=4000&pause=1000&color=00D9FF&center=true&vCenter=true&width=620&lines=%24+echo+%22Pretrain+once.+Deploy+everywhere.%22;%24+python+-c+%22import+life%3B+life.optimize()%22;%24+git+commit+-m+%22building+the+future%2C+one+model+at+a+time%22)](https://git.io/typing-svg)

</div>
