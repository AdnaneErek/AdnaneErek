<div align="center">

<!-- Typing Animation -->
[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=true&width=700&height=100&lines=Hi+%F0%9F%91%8B+I'm+Adnane+Erekraken;AI+Engineer+%7C+MSc+AI+%40+CentraleSupélec)](https://git.io/typing-svg)

<br/>

<!-- Social badges -->
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adnane-erekraken/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AdnaneErek)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:adnane.erekraken@student-cs.fr)

</div>

---

## 🧠 About Me

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
            "🔬 GPT-inspired Foundation Model for industrial PHM (CWRU, CMAPSS)",
            "🌟 POLARIS — AI-powered Strategic Steering & Portfolio Optimization Platform",
        ]

    def philosophy(self):
        return "Bridge cutting-edge AI research with real business impact."
```

---

## 🚀 Projects

### 🌟 POLARIS — Portfolio Optimization & Learning AI for Risk-Adjusted Strategy
> *An enterprise-grade AI platform for strategic decision-making — think of it as a North Star for steering committees.*

- **Monte Carlo** option evaluation with **CVaR10** risk-adjusted scoring
- Closed-loop **learning system**: outcome capture → forecast recalibration → scoring adaptation
- Executive-grade **PDF SteerCo reports**, interactive dashboards & conversational interface
- Built-in **guardrails**: budget, capacity, compliance & governance checks
- Stack: `Python` `Pandas` `scikit-learn` `Streamlit` `Plotly` `ReportLab` `Gemini API`

---

### ⚙️ Foundation Models for Fault Diagnosis & Prognosis — LGI Lab, CentraleSupélec
> *6-month research project (Oct 2025 – Mar 2026) · Team of 4 · HPC-scale training · Two parallel research tracks*

**The core problem:** Traditional PHM models fail to generalize — every new machine or fault type demands a fresh labeled dataset. This mirrors the pre-GPT era in NLP. The goal: build a GPT-inspired model that pretrained once, transfers everywhere.

<details>
<summary><b>🔍 Phase 1 — Data Leakage Investigation (critical methodological contribution)</b></summary>

<br/>

> Near-perfect accuracy figures (97–99.99%) in PHM literature looked suspicious. We dug in.

**Two systematic leakage sources identified & corrected:**

| Leakage Type | Root Cause | Fix |
|---|---|---|
| **Overlapping window leakage** | Overlapping train/test windows from the same signal → model memorizes waveforms | Non-overlapping windows + signal-level split |
| **Bearing-level leakage** | CWRU Drive End & Fan End sensors record the same bearing simultaneously, yet are split independently | Group-wise split by physical bearing identity |

**Quantified impact on MOMENT accuracy (CWRU):**

```
Standard protocol (both leakages)  →  98.97%  ████████████████████  
Fix bearing-level only             →  89.70%  ██████████████████    
Fix window-level only              →  83.46%  █████████████████     
Fully leak-free protocol           →  78.73%  ████████████████      
                                              ↑ 20pp gap = artifactual
```

> **Finding:** A large portion of reported SOTA improvements in PHM deep learning are artifactual. Group-wise splitting by physical unit identity should be the new standard.

</details>

<details>
<summary><b>🏗️ Phase 2 — Architecture Benchmarking & Fine-Tuning Strategy</b></summary>

<br/>

**Models benchmarked under leak-free protocol:**

| Model | Pretraining | Accuracy (leak-free) |
|---|---|---|
| GPT4TS | Time-series patches on GPT-2 backbone | — |
| **MOMENT** | Masked autoencoder on large TS corpus | **78.73%** → **97.38%** (multi-label) |
| NuTime | Transformer on UCR/UEA datasets | — |
| RoBERTa | Text pretrained, fine-tuned on TS | ~96% (leaky) |

**Task reformulation → multi-label classification:**
Rather than mutually exclusive fault classes, each sample gets a binary vector `y = [Ball, InnerRace, OuterRace]` — physically realistic, handles co-occurring faults, and reduces class imbalance. Result: **97.38% subset accuracy, Macro-AUC 0.9996** under leak-free protocol.

**Fine-tuning strategy that won — Progressive Unfreezing:**
```
Stage 1 → Linear probe     (encoder frozen, head only)
Stage 2 → Partial unfreeze (last transformer block)
Stage 3 → Full unfreeze    (entire encoder)
```
Preserves pretrained representations, avoids catastrophic forgetting.

</details>

<details>
<summary><b>🤖 Track 2 (Adnane) — Robot Fault Diagnosis & Sim-to-Real Transfer</b></summary>

<br/>

**Setup:** 4-DOF robot arm · 9 fault classes (healthy + 4 stuck + 4 steady-state error) · 9-channel trajectory signals (desired, realized, tracking error) · Only simulation data available for training, real robot for evaluation.

**The sim-to-real gap problem:**
```
Scratch training on sim data  →  81.94% sim  /  55.56% real  (gap: 26.4pp)
+ Generic Gaussian noise aug  →  sim ↑        /  48.89% real  (gap WIDENS)
```
Generic noise doesn't model the structured physical differences between sim and real.

**Physics-informed simulator augmentation (root cause analysis):**
- Actuation delays: sim used fixed 10-step delay → real robot has variable delays
- Healthy motor noise: sim was deterministic → real has Gaussian steady-state errors

**Fixes implemented in Simulink:**
```python
holdingTimeIdx  ~ Uniform(2, 30)          # variable actuation delay
normalMotorErr  ~ Gaussian(μ=0, σ≈1.2)   # discretized to [-3,...,3]
faultMagnitude  ~ Uniform(10, 50)         # expanded fault range
```

**Training strategy comparison (final results):**

| Strategy | Sim Accuracy | Real Accuracy | Gap |
|---|---|---|---|
| A1 — Scratch + Full FT (baseline) | 81.94% | 57.78% | 24.2pp |
| A2 — Scratch + Head Only | **90.19%** | 61.11% | 29.1pp |
| A3 — Supervised Pretrain + 25% unfreeze | 82.96% | 67.78% | 15.2pp |
| **A4 — Supervised Pretrain + 50% unfreeze** | 83.33% | **70.00%** | **13.3pp** |

> **Key insight:** High simulation accuracy ≠ good real-world performance. A2 hits 90.19% sim but only 61.11% real. Pretraining + partial unfreezing sacrifices sim accuracy but gains +12pp real accuracy — fitting the sim too tightly is **counterproductive** for sim-to-real transfer.

**Self-supervised objectives tested (both failed):**
```
Next-token prediction (GPT-style)   →  ~11%  (random chance)
Masked-token prediction (BERT-style) →  ~11%  (random chance)
```
Trajectory fault signatures require labeled pretraining — generic reconstruction losses don't learn fault-discriminative features.

</details>

<details>
<summary><b>🌐 Track 1 — General-Purpose Multi-Dataset Foundation Model</b></summary>

<br/>

**Datasets aggregated for pretraining:**
`CWRU` · `PRONOSTIA/FEMTO` · `CMAPSS (NASA)` · `MFPT` · *(gear & battery datasets in progress)*

**Objective:** Masked reconstruction pretraining across heterogeneous PHM domains — mechanical vibration → thermal degradation → aerodynamic performance.

**Key finding:** Multi-dataset pretrained model outperforms single-dataset pretraining on **held-out datasets not seen during pretraining** — early evidence of genuine cross-domain transfer.

**Scale challenge:** 12M+ rows of time-series data · ~2hrs/epoch on single GPU · Training on La Ruche HPC (SLURM)

</details>

**Stack:**
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![MATLAB](https://img.shields.io/badge/Simulink-0076A8?style=flat-square&logo=mathworks&logoColor=white)
![Linux](https://img.shields.io/badge/HPC_SLURM-FCC624?style=flat-square&logo=linux&logoColor=black)

---

### 🎵 Bach2BeethovenAI — Music Generation
> *BiLSTM trained on MIDI corpora to generate stylistically consistent classical compositions.*

- Tokenization via `music21`, temperature-based sampling, MIDI → WAV automation
- Stack: `PyTorch` `TensorFlow` `music21`

---

### 📈 LSTM Demand Forecasting — Marjane Holding
> *End-to-end supply chain forecasting system deployed in production.*

- LSTM-based demand forecasting reducing stock-outs across import planning
- Power BI dashboard centralizing cost, delay & seasonality KPIs
- Stack: `Python` `PyTorch` `Power BI` `SQL`

---

## 🛠️ Tech Stack

<div align="center">

### Languages & Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Data & Visualization
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![PowerBI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

### DevOps & Tools
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![GitLab](https://img.shields.io/badge/GitLab-FC6D26?style=for-the-badge&logo=gitlab&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![PowerShell](https://img.shields.io/badge/PowerShell-5391FE?style=for-the-badge&logo=powershell&logoColor=white)

### Platforms & Environments
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MATLAB](https://img.shields.io/badge/MATLAB-0076A8?style=for-the-badge&logo=mathworks&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## 📊 GitHub Stats

<div align="center">

<img height="180em" src="https://github-readme-stats.vercel.app/api?username=AdnaneErek&show_icons=true&theme=tokyonight&include_all_commits=true&count_private=true&hide_border=true"/>
<img height="180em" src="https://github-readme-stats.vercel.app/api/top-langs/?username=AdnaneErek&layout=compact&theme=tokyonight&hide_border=true"/>

</div>

<div align="center">

[![GitHub Streak](https://streak-stats.demolab.com/?user=AdnaneErek&theme=tokyonight&hide_border=true)](https://git.io/streak-stats)

</div>

---

## 🏆 Certifications

| Badge | Certification | Issuer |
|-------|--------------|--------|
| 📊 | Data Science Job Simulation | Boston Consulting Group |
| 💼 | Strategy Consulting Job Simulation | Boston Consulting Group |
| 📈 | Power BI Data Analyst Professional Certificate | Microsoft |
| 🗂️ | Google Project Management Specialization | Google |
| 💻 | Client Needs & Software Requirements | University of Alberta |

---

## 💬 Random Dev Quote

<div align="center">

[![Readme Quotes](https://quotes-github-readme.vercel.app/api?type=horizontal&theme=tokyonight)](https://github.com/piyushsuthar/github-readme-quotes)

</div>

---

## 📡 Contribution Activity

<div align="center">

[![Adnane's Activity Graph](https://github-readme-activity-graph.vercel.app/graph?username=AdnaneErek&theme=tokyo-night&hide_border=true&area=true&custom_title=Contribution%20Graph)](https://github.com/AdnaneErek)

</div>

---

## 🐍 Contribution Snake

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AdnaneErek/AdnaneErek/output/github-contribution-grid-snake-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/AdnaneErek/AdnaneErek/output/github-contribution-grid-snake.svg" />
  <img alt="github-snake" src="https://raw.githubusercontent.com/AdnaneErek/AdnaneErek/output/github-contribution-grid-snake-dark.svg" />
</picture>

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:00D9FF,100:0D1117&height=120&section=footer&animation=twinkling"/>

![Visitor Count](https://komarev.com/ghpvc/?username=AdnaneErek&color=00d9ff&style=for-the-badge&label=PROFILE+VIEWS)

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&duration=4000&pause=1000&color=00D9FF&center=true&vCenter=true&width=600&lines=Foundation+Models+%7C+Sim-to-Real+Transfer+%7C+Strategic+AI;Pretrain+once.+Deploy+everywhere.;Building+systems+that+think%2C+learn%2C+and+explain+themselves.)](https://git.io/typing-svg)

</div>
