# 🎙️ CausalSpectroNet: An Interpretable Pipeline for Parkinson’s Voice Analysis

This repository implements a complete, language‑independent workflow for **Parkinson’s Disease (PD) voice classification** and **acoustic interpretability**. We fuse spectrogram‑based causal modeling with symbolic acoustic tokens (via [WavTokenizer](https://github.com/xiaolaohu05/WavTokenizer)) to:

1. **Classify** HC vs. PD from speech
2. **Localize** discriminative time–frequency regions via Grad‑CAM  
3. **Extract** symbolic “acoustic tokens” and quantify their divergence  
4. **Reconstruct** listenable PD‑specific audio from masked spectrograms  

Unfortunately, I can't provide the raw database here due to copyrights, and the processed data was too large for Github to hold. Therefore, I strongly advice users to look up the database on the Internet and save them in a folder called 'Voice". Then run audio_preprocessing_pipeline.py. Please make sure those spectrograms are saved in the same level of the database folder, adding '_Output' at the end of the folder name.
---

## 🚀 Quick Start

1. **Clone this repo**  
   ```bash
   git clone https://your.repo.url/Parkinson.git
   cd Parkinson
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   # e.g. librosa, soundfile, torch, torchvision, transformers, scikit-learn, opencv-python, tqdm
   ```

3. **Prepare data**  
   - Place raw `.wav` files under `Voice/`.  
   - Directory structure will be created automatically:  
     ```
     Voice/
       ├── China_HC/… .wav
       └── China_PD/… .wav
     ```

4. **Run the full pipeline** (or invoke individual steps)  
   ```bash
   bash sleep_then_visualize.sh
   ```

---

## 🗂️ Directory Structure

```
.
├── audio_preprocessing_pipeline.py     # 1. WAV → 1 s segments + STFT/Mel/CQT PNG & metadata
├── batch_gradcam_and_topk.py           # Batch GradCAM + Top‑K spectrogram pairing
├── batch_gradcam_spectro_*.py          # Per‑variant GradCAM (ConstantQ / GlobalLocal / CausalAtt / Scalogram)
├── generate_discriminative_audio.py    # Spectrogram → masked spectrogram → Griffin‑Lim → .wav
├── gradcam_summary_figure.py           # Plot average HC/PD GradCAM for each spectrogram type
├── gradcam_token_overlay_pipeline.py   # Overlay GradCAM + acoustic‐token heatmap on Top‑K pairs
├── run_all_experiments.py              # Orchestrates ablations, training, GradCAM, token analysis
├── spectro_ablation_experiments.py     # Compare CausalAtt vs. ConstantQOnly vs. GlobalLocalOnly vs. ScalogramOnly
├── spectro_ablation_experiments_single.py # Single‑variant ablation driver
├── spectrogram_model_gradcam.py        # Helper: GradCAM on a single spectrogram image
├── token_spectrogram_pipeline.py       # 2. Tokenize spectrograms via WavTokenizer → `audio_token_results.csv`
├── train_causal_spectro_model.py       # 3. Train CausalSpectroNet (global‑local perturbations + causal loss)
├── token_model_extensions.py           # Helper extensions for acoustic token clustering & analysis
├── token_model_training.py             # (Optional) Fine‑tune or train your own WavTokenizer variant
├── multi_modal_training.py             # (Optional) Train a joint audio + text (annotations) classifier
├── run_all_experiments.py              # Top‑level orchestration of all steps above
│
├── ablation/                           # Ablation scripts & configs
├── figures/                            # Saved PNGs: spectrograms, GradCAM maps, overlays
├── logs/                               # Training & experiment logs (seed0, etc.)
├── models/                             # Model definitions & utilities
├── result_data/                        # CSVs of results: `ablation_results_*.csv`, `token_topk_scores.csv`, …
├── saved_models/                       # Checkpoints: ConstantQOnly_seed0.pth, CausalAtt_seed0.pth, …
├── training/                           # Training pipelines for token & multimodal models
│   ├── token_model_training.py
│   ├── token_model_extensions.py
│   └── multi_modal_training.py
├── Voice/                              # Input audio + auto‑generated `*_Output/` directories
│   ├── China_HC/
│   └── China_PD/
└── spectrum.ipynb, Token.ipynb         # Exploratory notebooks (code snapshot)
```

---

## 🔧 Step‑by‑Step Workflow

### 1. Audio Preprocessing → Spectrograms
```bash
python audio_preprocessing_pipeline.py
```
- Splits each `.wav` into 1 s segments (silence removal via `librosa.effects.split`).  
- Generates **STFT**, **Mel**, **CQT** spectrograms (dB → color PNG via Matplotlib).  
- Saves PNG and metadata `.json` under `Voice/..._Output/`.

### 2. Tokenization via WavTokenizer  
```bash
python token_spectrogram_pipeline.py
```
- **Refer to** [WavTokenizer](https://github.com/xiaolaohu05/WavTokenizer) for installing & configuring the tokenizer.  
- Takes spectrogram PNGs, extracts discrete tokens (e.g. 64‑codebook), outputs `audio_token_results.csv`.

### 3. Train CausalSpectroNet  
```bash
python train_causal_spectro_model.py
```
- **Global‑Local Transform**: FFT‐mask + random erasing.  
- **CausalSpectroNet** (ResNet18 backbone):  
  - Dice loss for attention‐map consistency  
  - Symmetric KL loss for predictive consistency  
- Produces checkpoints in `saved_models/`.

### 4. Ablation Studies  
```bash
python spectro_ablation_experiments.py
```
- Compare four variants:  
  1. ConstantQOnly  
  2. GlobalLocalOnly  
  3. CausalAtt (full model)  
  4. ScalogramOnly  
- Outputs `ablation_results_*.csv` in `result_data/`.

### 5. GradCAM Visualizations  
#### a. Average Heatmaps  
```bash
python gradcam_summary_figure.py
```
- Computes & plots mean GradCAM over all HC vs. PD samples for each spectrogram type.

#### b. Top‑K Pair Overlays  
```bash
python gradcam_token_overlay_pipeline.py
```
- Finds Top‑K most divergent HC/PD spectrogram pairs.  
- Generates GradCAM + token‐sequence overlay under `figures/gradcam_token_overlay/`.

### 6. Token Divergence Analysis  
```bash
python kl_js_token_analysis.py
```
- Reads `audio_token_results.csv`.  
- Computes **symmetric KL** & **Jensen‑Shannon** divergence over codebook tokens.  
- Lists Top‑K discriminative tokens and saves `token_topk_scores.csv`.

### 7. Discriminative Audio Reconstruction  
```bash
python generate_discriminative_audio.py
```
- Restores magnitude spectrogram from PNG, masks with GradCAM heatmap.  
- Inverts to waveform via Griffin‑Lim → `discriminative_pd.wav`, `discriminative_hc.wav`.  
- Playable in Jupyter / saved to disk.

---

## 📈 Results & Interpretation

- **Ablation**: Full CausalAtt model consistently outperforms single‐variant baselines.  
- **GradCAM**: PD samples show focused activations in low‐frequency tremor bands & phrase ends.  
- **Token KL**: Token IDs `3, 60, …` exhibit highest KL divergence (PD vs. HC).  
- **Audio**: Reconstructed PD waveform highlights high‐variance tremor frequencies.

---

## 🤝 Acknowledgements & Citation

- **Grad‑CAM**: *Selvaraju et al., ICCV 2017*.  
- **WavTokenizer**: *Ji et al., ICLR 2025*.  
- **Parkinson’s vocalization dataset**: *Apaydin et al.*  

