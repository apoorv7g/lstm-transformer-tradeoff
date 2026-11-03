# Literature Review

Each paper from the proposal is listed with what it found and how it supports this study. Status
flags:
- **[verified]**: the paper was located and its abstract or content checked (Oct 2026)
- **[partly verified]**: the bibliographic record was found, but the findings were not checked
- **[unverified]**: not located under the given title. Get the DOI before citing

> Before the final report, every [partly verified] and [unverified] item needs a human check against the PDF. Do not copy
> numbers from this file into the paper without checking them first.

---

## A. Papers cited in the proposal

### 1. [verified] Owais, Arsalan, Choi, Mahmood, Park (2019). *Artificial Intelligence-Based Classification of Multiple Gastrointestinal Diseases Using Endoscopy Videos for Clinical Diagnosis.* J. Clin. Med. 8(7):986. doi:10.3390/jcm8070986. [PMC6678612](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6678612/)
- Two cascaded ResNets give spatial features, and an LSTM gives temporal features. Multiple GI
  disease classes, about 52k frames. Reported AUC ≈ 97%.
- **Relevance:** a CNN→LSTM medical-video precedent that reports model size. It supports the LSTM
  arm.
- *Note:* the full title is longer than the proposal slide's, so update the citation.

### 2. [verified] Barros, Lacerda, Albuquerque, Conci (2021). *Pulmonary COVID-19: Learning Spatiotemporal Features Combining CNN and LSTM Networks for Lung Ultrasound Video Classification.* Sensors 21(16):5486. doi:10.3390/s21165486. [PMC8401701](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8401701/)
- 185 lung-ultrasound videos. Best model: ImageNet Xception → LSTM(512), dropout 0.4, 2 × FC-1024,
  **20-frame sequences**, tuned with Optuna. About 93% mean accuracy and 97% COVID sensitivity,
  beating spatial-only models.
- **Relevance:** **the closest analogue to our setup**: a small medical video set, frozen ImageNet
  CNN + LSTM, and T = 20. It is evidence that LSTMs do well in the small-data regime, so it is a
  pillar of H₀.

### 3. [unverified] *Towards Accurate Surgical Workflow Recognition with Convolutional Networks and Transformers.*
- Not found under this exact title. It may be Kadkhodamohammadi et al., *Computer Methods in
  Biomechanics and Biomedical Engineering: Imaging & Visualization* (≈ 2021–22). Confirm the DOI.
- The proposal slide claims it evaluates convolutional, Transformer and hybrid designs, with 93%
  frame-level accuracy for the hybrid. **Unverified.**
- Verified substitute if needed: **Trans-SVNet** (Gao et al., MICCAI 2021, arXiv:2103.09712), which
  fuses CNN spatial and temporal embeddings with a Transformer for surgical phase recognition.

### 4. [verified] Soeleman et al. (2022). *Video Violence Detection Using LSTM and Transformer Networks Through Grid Search-Based Hyperparameters Optimization.* Int. J. Safety & Security Eng. (IIETA) 12(5):615–622. doi:10.18280/ijsse.120510. [iieta.org/node/12025](https://www.iieta.org/node/12025)
- InceptionV3 per-frame features feed an LSTM or a Transformer, each tuned by grid search.
  **The LSTM beat the Transformer** on Hockey, Crowd and AIRTLab, with AUC up to 0.976, 0.934 and
  0.86.
- **Relevance:** a direct LSTM-vs-Transformer comparison on frozen CNN features with small video
  datasets. Its design matches ours. It is **the strongest grounding for H₀**.

### 5. [partly verified] Van-Deste, Castro Lopes, Lopes (2025). *Memory vs. Attention: Investigating LSTM and Transformer Models in Human Action Recognition.* Procedia Computer Science 278:1299–1306 (CENTERIS 2025). [researchr record](https://researchr.org/publication/Van-DesteLL25)
- A direct comparison of LSTM, BiLSTM and Transformer on skeleton-based action recognition.
- **The venue differs from the proposal**, which says IBIMA. The record found says
  CENTERIS / Procedia, so check it. **Findings unverified** (no abstract available).

### 6. [verified] Alomar, Aysel, Cai (2024/2025). *CNNs, RNNs and Transformers in Human Action Recognition: A Survey and a Hybrid Model.* arXiv:2407.06162; Artificial Intelligence Review 58:387 (2025), doi:10.1007/s10462-025-11388-3.
- Surveys CNN, RNN and ViT approaches to human action recognition and proposes a hybrid CNN–ViT.
- **Relevance:** the survey backdrop for the three families, and support for the hybrid idea.

### 7. [verified] Czempiel, Paschali, Ostler, Kim, Busam, Navab (2021). *OperA: Attention-Regularized Transformers for Surgical Phase Recognition.* MICCAI 2021, LNCS 12904:604–614. doi:10.1007/978-3-030-87202-1_58. arXiv:2103.03873.
- A Transformer over long sequences of CNN frame features, with an attention-regularisation loss
  that steers attention toward high-quality frames. It beats temporal-refinement baselines on two
  cholecystectomy datasets.
- **Relevance:** Transformers help on **long** surgical videos (thousands of frames). That is the
  opposite regime to our short clips. The attention-map visualisation idea is borrowed for our
  qualitative figure.

### 8. [partly verified] Pan, Gao, Wang, Zhang, Mu, He (2022/23). *Temporal-based Swin Transformer Network for Workflow Recognition of Surgical Video.* IJCARS 18:139–147.
- TSTNet: Swin Transformer for spatial features, an **LSTM** for temporal modelling, and a
  prior-revision step using phase-order knowledge. Evaluated on Cholec80. Accuracy figures
  **unverified**.
- **Relevance:** a "Transformer spatial + LSTM temporal" hybrid, which shows the LSTM is still
  chosen as the temporal model in recent work.

### 9. [verified] Mathai, Liu, Ling (2024). *A Hybrid Transformer-LSTM Model With 3D Separable Convolution for Video Prediction.* IEEE Access 12:39589–39602. [PDF](https://www.cse.scu.edu/~yliu1/papers/A_Hybrid_Transformer-LSTM_Model_With_3D_Separable_Convolution_for_Video_Prediction.pdf)
- 3DTransLSTM interleaves self-attention and recurrence, with 3D depthwise-separable convolutions.
  Competitive with far fewer parameters and less compute ("Green AI").
- **Relevance:** precedent for a compute-aware hybrid (our stretch goal). *Caveat:* the task is
  video **prediction**, not classification.

### 10. [unverified] *The Discussion About Suitable Scenarios for BiLSTM and Transformer.*
- Not found under this title. It is possibly a small proceedings paper (Atlantis Press /
  *Highlights in Science, Engineering and Technology*?). Get the PDF or DOI from whoever added it.
- The proposal claims it compares FPS, FLOPs and training time. **Unverified.**
- Related, verified evidence to cite instead or as well:
  - **LREC 2022**, *Transformer versus LSTM Language Models trained on Uncertain ASR Hypotheses in
    Limited Data Scenarios* ([aclanthology 2022.lrec-1.41](https://aclanthology.org/2022.lrec-1.41)).
    The Transformer gained on the larger corpus, but was **on par with the LSTM on the smaller
    one**.

## B. Background references (architecture and methods)

| Reference | Use in this project |
|---|---|
| Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* | LSTM cell definition |
| Vaswani et al. (2017), *Attention Is All You Need* | Transformer encoder, sinusoidal PE |
| Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture* | Justifies **pre-LN** |
| Sandler et al. (2018), *MobileNetV2* | Frozen backbone; ≈ 300 M MACs per 224² frame, ≈ 3.4 M params (with classifier) |
| Arnab et al. (2021), *ViViT*, arXiv:2103.15691 | "Factorised encoder": a spatial encoder per frame, then a temporal Transformer. Our design is this with a frozen CNN |
| Bertasius et al. (2021), *TimeSformer*, arXiv:2102.05095 | Divided space-time attention. Context only (too heavy for CPU) |
| Neimark et al. (2021), *Video Transformer Network*, arXiv:2102.00719 | 2D backbone + temporal Transformer |
| Gao et al. (2021), *Trans-SVNet*, arXiv:2103.09712 | CNN features + Transformer, surgical phase |
| Sayak Paul, Keras example *Video Classification with Transformers* ([keras.io](https://keras.io/examples/vision/video_transformers/)) | Code template. DenseNet121 features, T = 20, learned PE, 1 post-LN block, max-pool. About 4.23 M params, ≈ 89% on 5 UCF101 classes. Notes it "works best with a larger dataset" |
| Feng, Tung, Ahmed, Bengio, Hajimirsadeghi (2024), *Were RNNs All We Needed?*, arXiv:2410.01201 | minLSTM / minGRU: gates without h_{t−1}, parallel-scan trainable. Optional cheap baseline |
| Nadeau & Bengio (2003), *Inference for the Generalization Error* | Corrected resampled t-test |
| Bouckaert & Frank (2004), *Evaluating the Replicability of Significance Tests* | Corrected repeated k-fold t-test |
| Dietterich (1998), *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms* | 5×2cv test, and why naive k-fold t-tests mislead |
| DeLong et al. (1988) | AUROC comparison on paired predictions |

Background references were cited from memory and should be checked before the paper.

## C. Synthesis: what the literature predicts for us

1. **Large data, long sequences → Transformers win** (ViViT, TimeSformer, OperA on long surgical
   video).
2. **Small data, short sequences, frozen features → LSTM ≥ Transformer, or parity**: Soeleman 2022
   (LSTM won), Barros 2021 (LSTM strong on 185 videos), the Keras example's own caveat, and
   LREC 2022 (parity in limited data).
3. **Our regime is clearly (2):** 535 clips, T ≤ 40, frozen MobileNetV2. So **H₀ (no accuracy
   advantage for the Transformer) is the literature-consistent null.** Rejecting it would be the
   surprising result.
4. **Compute claims in the literature are mostly GPU-based or FLOP-based.** Few papers report
   batch-1 CPU latency. Our contribution on RQ2 fills that gap, and our analysis
   (`transformer_design.md` §5) predicts the "Transformers are slower" folk wisdom may not hold at
   T ≤ 40 on CPU.
