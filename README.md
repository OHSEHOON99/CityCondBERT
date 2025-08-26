# SIGSPATIAL 2025 GIS Cup (HuMob Challenge) – SCSI Lab, Yonsei University

**Team:** SCSI Lab (Yonsei University, Dept. of Civil & Environmental Engineering)  
**Lead Contributor:** **Sehoon Oh** (Master’s Student)  
**Track:** 2025 GIS Cup – HuMob Challenge, Track 2 (Phase 1)

> **Current best (City A)**: **GEO-BLEU 0.1733** (test)

---

## 1. Overview

Our system builds on a strong baseline and adds:  
1. **Multi-modal feature embeddings** (categorical / periodic / learnable Fourier) across **spatio-temporal** features.  
2. A new **differentiable training objective** `GeoBleuSinkhornLoss`, aligning predicted vs. reference trajectories using **entropy-regularized optimal transport** in an n-gram–like manner.  
3. A **BERT-style encoder** that treats mobility prediction as **masked sequence modeling**, with an option to extend to **sequence-to-sequence forecasting** for future trajectories.

---

## 2. Baseline & Borrowed Components (Attributions)

- **Backbone**: **ST-MoE-BERT**  
  He, H., Luo, H., & Wang, Q. R. (2024). *ST-MoE-BERT: A Spatial-Temporal Mixture-of-Experts Framework for Long-Term Cross-City Mobility Prediction.* In Proc. of the 2nd ACM SIGSPATIAL Workshop on Human Mobility Prediction Challenge (pp. 10-15).  
  [GitHub](https://github.com/he-h/ST-MoE-BERT/tree/main)

- **Learnable Fourier Features (continuous signals)**  
  Li, Y., Si, S., Li, G., Hsieh, C. J., & Bengio, S. (2021). *Learnable Fourier features for multi-dimensional spatial positional encoding.* NeurIPS 34, 15816-15829.

- **Periodic Encodings (temporal cyclicity)**  
  Wu, X., He, H., Wang, Y., & Wang, Q. (2024). *Pretrained mobility transformer: A foundation model for human mobility.* arXiv:2406.02578.

We **explicitly acknowledge** the above works; our code **adapts** these ideas and integrates them into a unified **FeatureBlock** and training pipeline.

---

## 3. Key Contributions

### 3.1 Multi-Modal Feature Embeddings

**Why multiple representations?**  
Human mobility features are heterogeneous: `location` is inherently **discrete**, `time/dow` are **periodic** (periods: **48** for time-of-day, **7** for day-of-week), and `delta` behaves like a **continuous** signal with long-tailed scales (e.g., inter-step displacement or time gap). Using a single representation introduces an unfavorable inductive bias. By combining **categorical**, **periodic**, and **learnable Fourier** encodings, we preserve (i) discrete vocab structure for grids, (ii) cyclic proximity for temporal features, and (iii) smooth/scale-robust behaviors for continuous deltas—leading to better generalization on sparse cells and more stable long-horizon forecasts.

**How it plugs into the model (BERT-front embedding):**  
At each timestep, per-feature encoders produce mode-specific embeddings (categorical / periodic / Fourier). We apply an **inner combine** rule (`cat` | `sum` | `mlp`) within each feature, then a **feature fusion (outer combine)** across features to form a single token embedding $\mathbf{x}_t \in \mathbb{R}^{d_{model}}$. This token sequence is fed to the **BERT-style encoder**, optionally with positional embeddings. LayerNorm/Dropout after fusion stabilize training.

**Config-driven experimentation:**  
Embedding choices are exposed in `config.py` for ablations and reproducibility:

```python
def default_feature_configs():
    return {
        "day":      {"modes": ["categorical"],                 "combine_mode_inner": "cat"},
        "time":     {"modes": ["categorical", "periodic"],     "combine_mode_inner": "cat", "period": 48},
        "dow":      {"modes": ["categorical", "periodic"],     "combine_mode_inner": "cat", "period": 7},
        "weekday":  {"modes": ["categorical"],                 "combine_mode_inner": "cat"},
        "location": {"modes": ["categorical"],                 "combine_mode_inner": "cat"},
        "delta":    {"modes": ["fourier"],                     "combine_mode_inner": "cat"}
    }
```

---

### 3.2 `GeoBleuSinkhornLoss`

**Why we needed a new loss:**  
Most existing baselines treat mobility prediction as **classification over 40,000 grid cells** using **CrossEntropy (CE)**. CE penalizes any non-exact prediction equally, ignores **spatial similarity**, and lacks **sequence context**.

**What GEO-BLEU brings (and the gap):**  
- **Spatial similarity:** distance decay rewards nearby predictions.  
- **Sequential matching:** **n-gram precision** evaluates multi-step patterns.  
- **But:** Classic GEO-BLEU is **non-differentiable** (eval-only).

**Our approach: make GEO-BLEU trainable via OT (Sinkhorn).**  
1. **Local offsets & kernel weighting:** build a local window (e.g., 7×7) around each cell and compute distance-weighted similarities.  
2. **n-gram similarity matrices:** for each n (1–5), slide over sequences to form predicted vs. true n-gram sets.  
3. **Sinkhorn OT alignment:** treat n-gram sets as distributions; apply **entropy-regularized OT** with **log-domain Sinkhorn** to obtain a soft, differentiable alignment score.  
4. **Loss aggregation:** weighted precision across n=1..5, combined with CE:  
   Weighted precision across n = 1..5, combined with CE:

   $$
   L = \\alpha \\cdot CE + (1 - \\alpha) \\cdot \\big(1 - \\text{GeoBLEU}\\big)
   $$

   Start CE-heavy, then anneal $\alpha$ to emphasize GeoBLEU.

**Key hyperparameters:**  
```python
"combo": {
    "ce_name": "ce",
    "geobleu_kwargs": {
        "H": 200, "W": 200,
        "n_list": [1, 2, 3, 4, 5],
        "win": 7,
        "beta": 0.5,
        "cell_km_x": 0.5, "cell_km_y": 0.5,
        "distance_scale": 2.0,
        "eps": 0.1,
        "n_iters": 30
    },
    "alpha_warmup_epochs": 10,
    "alpha_transition_epochs": 20
}
```

---

## 4. Running

Our pipeline is driven by **argparse** with **best-known defaults in `config.py`**.  
- CLI flags can override `config.py` at runtime.

**Commands:**
```bash
# Train: full loop (train + validation) and final masked prediction
python main.py --mode train --city A

# Predict: load a saved checkpoint and run masked prediction only
python main.py --mode predict --city A --model_name <run_name>
```

**Checkpoint structure (per run under `checkpoints/`):**
```
{model_name}-{YYYYMMDD}_{HHMMSS}/
├── config.json     # snapshot of parameters at training start; reused at predict time
├── run_meta.json   # run metadata (seed, host, timings)
├── train_log.txt   # training/validation logs
└── results/        # predictions, evaluation outputs
```

---

## 5. Tested Environments

| Environment | OS / Kernel | GPUs | Driver / CUDA |
|---|---|---|---|
| **Env 1 (5090×4)** | Ubuntu 25.04 (GNU/Linux 6.14.0-15-generic x86_64) | NVIDIA 5090 ×4 | Driver **570.133.07** / CUDA **12.8** |
| **Env 2 (A5000×4)** | Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-139-generic x86_64) | NVIDIA A5000 ×4 | Driver **470.256.02** / CUDA **11.4** |

---

## 6. Installation (Conda)

For exact reproduction, use the provided **Conda explicit specs** and choose the file matching your machine:

- **5090×4 (CUDA 12.8)** → `requirements_5090.txt`  
- **A5000×4 (CUDA 11.4)** → `requirements_A5000.txt`

**Example:**
```bash
conda create -n env_name --file requirements_*.txt
conda activate env_name
```
> Replace `requirements_*.txt` with the correct file for your machine.

---

## 7. Citations

- He et al. (2024) – ST-MoE-BERT  
- Li et al. (2021) – Learnable Fourier Features  
- Wu et al. (2024) – Pretrained Mobility Transformer