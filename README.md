# IAB Content Taxonomy Classification via LLM Knowledge Distillation

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)


> Developed and trained entirely on a MacBook M4 Max (64GB RAM, MPS acceleration). The full pipeline -- from LLM data correction (97K domains) through embedding generation and model training -- completes in under 5 hours. The LLM labeling runs asynchronously via AWS Bedrock while all model training (TF-IDF, MLP distillation, ModernBERT fine-tuning) fits within Apple Silicon's memory budget. This is a compute-constrained study on a 98K-domain dataset -- at production scale with multi-GPU training, larger embedding models, and millions of domains, the dense approaches (distilled MLP, fine-tuned transformers) are expected to significantly outperform sparse baselines.

## What This Project Is

Production URL classification in adtech has historically relied on sparse NLP pipelines -- TF-IDF over crawler-extracted keywords from page content, trained on programmatically generated features. That architecture (advanced web crawlers feeding keyword extraction into TF-IDF + SVM classifiers) shipped in 2018 and still powers real-time bidding decisions at scale. **This project asks: what changes when you replace the crawler-based keyword extraction with frontier LLMs as the feature generation engine, and can modern dense representations (transformer encoders, knowledge distillation from billion-parameter teachers) outperform the classical sparse pipeline when scaled beyond this research dataset?**

On this dataset and compute budget, fine-tuned ModernBERT (91.7%) and TF-IDF + LinearSVC (91.6%) achieve near-identical top-1 accuracy -- the transformer matches the sparse baseline even on just 78K training domains. With production-scale training data (millions of domains), higher-capacity encoders (GTE-ModernBERT 768-dim, Matryoshka representations), and proper GPU infrastructure for exhaustive hyperparameter search, fine-tuned transformers are expected to surpass the sparse ceiling. The project demonstrates the full pipeline architecture that enables that scaling.

We implement a complete **LLM-as-data-engine pipeline** on the [Kaggle Website IAB Categorization](https://www.kaggle.com/datasets/bpmtips/websiteiabcategorization) dataset (98K domains, 27 IAB Tier-1 categories) -- programmatic label synthesis, soft-label knowledge distillation, sentence-transformer embedding generation, and multi-architecture downstream training. The pipeline discovers that **49.7% of crowd-sourced annotations are incorrect**, corrects them via LLM consensus, and then benchmarks four fundamentally different model architectures on the cleaned signal:

1. **TF-IDF + LinearSVC** -- sparse high-dimensional projection (30K features) with max-margin classification on LLM-generated keyword vocabularies
2. **MLP + KL Distillation** -- dense 384-dim E5-small-v2 embeddings distilled from Opus soft probability distributions via temperature-scaled KL divergence (T=2.0, alpha=0.7)
3. **ModernBERT Fine-tuning** -- end-to-end 150M-parameter encoder adaptation with discriminative pre-training on corrected supervision
4. **Linear baselines** (Logistic Regression, SGD) -- convex optimization on identical TF-IDF features to isolate the contribution of non-linear decision boundaries

## Observations at This Scale (98K Domains, Single-Machine Training)

On this dataset, fine-tuned ModernBERT (91.7%) and TF-IDF + LinearSVC (91.6%) achieve near-identical top-1 accuracy, both outperforming the distilled MLP (84.9%) by ~6.7 percentage points. The transformer matches the sparse baseline even at this constrained scale -- with more data and compute, it is expected to pull ahead.

The mechanism at this scale is information-theoretic. When an LLM generates keywords like "online store, electronics, gadgets" for an e-commerce domain, those tokens are **already category-aligned features** -- they carry near-maximal mutual information with the target label. TF-IDF with sublinear term frequency and inverse document frequency weighting preserves this discriminative structure: "automotive" receives high IDF weight and maps almost bijectively to "Autos & Vehicles." The LinearSVC's max-margin hyperplane in this 30K-dimensional space (99.9% sparsity, ~70 non-zero features per document) exploits the near-linear separability that the LLM created.

Dense encoders (E5-small-v2, 384-dim) perform dimensionality reduction that compresses lexical precision into a continuous space. In the dense embedding space, "automotive" and "car dealership" and "vehicle financing" collapse into a neighborhood -- useful for semantic similarity retrieval, but at this data scale it trades classification-relevant variance for semantic smoothness. The MLP must then learn non-linear decision boundaries to recover discriminative structure.

**Why dense methods are expected to win at scale:** The E5-small-v2 encoder used here (33M params, 384-dim) is a relatively small embedding model. With larger encoders (GTE-ModernBERT 768-dim, Matryoshka representations), millions of training domains (vs 78K), multi-GPU fine-tuning with exhaustive hyperparameter search, and richer text signals (full page content rather than domain+keywords), dense representations capture compositional semantics and handle edge cases that bag-of-words fundamentally cannot. The MLP distillation and ModernBERT fine-tuning pipelines implemented here are the architectures that scale -- this project validates the end-to-end machinery at research scale before production deployment.

## The Two-LLM Architecture: Decomposed Annotation Pipeline

A factored approach that separates deterministic classification (hard labels, keyword extraction) from probabilistic calibration (soft distributional targets). Each LLM operates at a different cost-quality tradeoff point, with the cheaper model handling exhaustive coverage and the expensive model providing rich distillation signal on a stratified subset:

```
+==============================================================================+
|  SONNET 4 (Label Corrector + Keyword Generator)                              |
|==============================================================================|
|  Coverage: ALL 97K domains                                                   |
|  Output:   Corrected tier1 category + 5-10 English keywords per domain       |
|  Purpose:  Fix 49.7% wrong Kaggle labels; generate semantic text features    |
|  Speed:    ~7 domains/sec (10 async workers via AWS Bedrock)                 |
|  Cost:     Lower (Sonnet pricing, straightforward classification task)       |
+==============================================================================+

+==============================================================================+
|  OPUS 4.6 (Soft Label Teacher)                                               |
|==============================================================================|
|  Coverage: 40K+ stratified subset                                            |
|  Output:   Probability distribution over all 27 categories per domain        |
|  Purpose:  Distillation signal -- soft labels encode inter-class structure    |
|  Speed:    ~2.5 domains/sec (5 async workers via AWS Bedrock)                |
|  Cost:     Higher (Opus pricing, nuanced probabilistic judgment)             |
+==============================================================================+
```

### Data Flow for a Single Domain

```
"espn.com" (Kaggle label: "Arts & Entertainment" -- WRONG)
     |
     |--- Sonnet 4 corrects   --> tier1 = "Sports"
     |                            keywords = "football, NBA, MLB, scores, highlights"
     |
     |--- Opus 4.6 produces   --> {Sports: 0.95, Entertainment: 0.15, Games: 0.08, ...}
     |
     |--- TF-IDF vectorizes   --> sparse 30K-dim vector (keywords become features)
     |--- E5-small encodes    --> dense 384-dim embedding
     |
     |--- LinearSVC classifies from TF-IDF --> "Sports" (0.02ms)
     |--- MLP classifies from embedding    --> "Sports" (1ms with cached embedding)
     |--- ModernBERT classifies end-to-end --> "Sports" (10ms)
```

## Results

### Model Comparison (all evaluated on the same corrected validation set: 9,810 domains)

| Model | Val Top-1 | Val Top-3 | Test Top-1 | Macro-F1 | Params | Inference |
|-------|-----------|-----------|------------|----------|--------|-----------|
| **TF-IDF + LinearSVC** | **91.6%** | **99.3%** | **91.7%** | **0.880** | ~30K features | 0.02 ms |
| **ModernBERT (corrected labels)** | **91.7%** | **99.2%** | -- | -- | 150M | ~10 ms |
| TF-IDF + Logistic Regression | 90.4% | 99.4% | -- | 0.878 | ~30K features | 0.02 ms |
| MLP (E5 + KL distillation) | 84.9% | 98.3% | -- | 0.814 | 337K | ~1 ms |

### v1 Baseline (noisy Kaggle labels, for comparison)

| Model | Val Top-1 | Val Top-3 | Params |
|-------|-----------|-----------|--------|
| v1 ModernBERT | 61.3% | 80.0% | 150M |
| v1 MLP (distilled) | 45.1% | 68.0% | 135K |
| Random baseline | 3.7% | 11.1% | -- |

### The Impact of Label Correction (v1 to v2): Supervision Quality vs Model Capacity

| Factor | v1 (noisy labels) | v2 (corrected) | Delta |
|--------|-------------------|----------------|-------|
| ModernBERT Top-1 accuracy | 61.3% | 91.7% | +49.6% relative |
| MLP Top-1 accuracy | 45.1% | 84.9% | +88% relative |
| MLP Top-3 accuracy | 68.0% | 98.3% | +45% relative |
| Teacher coverage | 8.4% (6.5K domains) | 42.4% (32.9K domains) | 5.1x |
| Inter-annotator agreement (LLM-LLM) | 50.9% Kaggle-Opus | 81.7% Sonnet-Opus | +60% |

The v2 ModernBERT (91.7%) trained on corrected labels improves by 30.4 percentage points over its v1 counterpart (61.3%) trained on noisy Kaggle labels -- same architecture, same hyperparameters, only the supervision quality changed. Even the tiny v2 MLP (337K params, 1.3 MB) outperforms the v1 ModernBERT (150M params, 602 MB) by 23.6 percentage points -- confirming that label quality dominates model capacity for classification.

### Architectural Analysis

**Sparse projections preserve LLM-injected discriminative signal at small scale.** When the upstream feature generator (Sonnet) produces tokens conditioned on category membership, TF-IDF with sublinear TF (1 + log(tf)) and IDF weighting creates a feature space where category-indicative tokens receive near-orthogonal representation. The LinearSVC finds max-margin separating hyperplanes in this high-dimensional space with 99.9% sparsity. At 98K domains, this feature space is sufficient -- at millions of domains with long-tail categories and ambiguous domains, the fixed vocabulary becomes a limitation.

**Dense sentence embeddings trade lexical precision for compositional generalization.** E5-small-v2 maps variable-length keyword sequences into a fixed 384-dimensional L2-normalized hypersphere. This projection optimizes for contrastive similarity (InfoNCE loss during pre-training) -- tokens with related semantics cluster together. At this scale, the compression costs more than it gains. However, with larger encoders (768-dim+), domain-adaptive pre-training, and production-scale fine-tuning data, dense representations capture compositional patterns and handle novel domains that sparse methods cannot generalize to.

**The distillation pipeline architecture scales independently of this dataset.** Temperature-scaled KL divergence (T=2.0) from Opus soft labels provides rich gradient signal -- the teacher's probability mass over 27 categories encodes inter-class similarity structure (e.g., "Sports" and "Games" co-receive probability mass). The alpha=0.7 weighting of KL vs BCE reflects that soft labels are more informative than hard labels. With larger embedding models and more teacher-labeled data, this distillation signal is expected to push the MLP well beyond the TF-IDF ceiling.

**ModernBERT fine-tuning represents the production-scale architecture.** End-to-end encoder fine-tuning (150M params) with discriminative learning rates avoids the two-stage embedding bottleneck entirely. The encoder learns task-specific representations directly. On this 78K-domain training set, the model is data-limited relative to its capacity -- at production scale (millions of labeled domains, multi-GPU training), fine-tuned transformers typically dominate all other approaches for text classification.

**Hyperparameter sensitivity of TF-IDF confirms its simplicity advantage at small scale.** Grid search over configurations (20K-50K features, unigrams through trigrams, various min_df thresholds) shows only 1.4pp total spread -- the method is robust but also has a low ceiling that neural approaches can exceed with sufficient data and compute.

**Generalization metrics confirm no distributional shift.** Val/test gap of 0.1pp (91.6% vs 91.7%) indicates the Sonnet-generated keywords have consistent lexical distributions across splits.

### Per-Category Performance (TF-IDF + LinearSVC)

Strong performers (F1 > 0.95): Adult (0.963), Travel (0.962), Real Estate (0.958), Jobs & Education (0.955), News (0.954), Autos & Vehicles (0.953), Finance (0.950), Health (0.950)

Weak performers: Sensitive Subjects (0.000 -- only 3 val samples), Hobbies & Leisure (0.812), Online Communities (0.833), Science (0.854)

## Key Findings

1. **Annotation quality dominates model capacity by an order of magnitude.** v2 ModernBERT (corrected labels) achieves 91.7% vs v1 ModernBERT (noisy labels) at 61.3% -- a 30.4pp improvement from identical architecture with only label quality changed. Even a 337K-param MLP with corrected labels (84.9%) outperforms the 150M-param v1 ModernBERT by 23.6pp. The Bayes error rate for classification is set by label noise, not model expressiveness.

2. **Crowd-sourced taxonomic annotations exhibit catastrophic error rates.** 49.7% of Kaggle labels are incorrect under LLM consensus (Sonnet-Opus inter-annotator agreement: 81.7% vs Kaggle-Opus: 50.9%). The error distribution is systematic, not random: Shopping was undercounted by 104% (e-commerce domains misclassified as Home & Garden, Business & Industrial), and Sensitive Subjects had 99% label error (annotator avoidance bias). This level of systematic noise cannot be overcome by regularization or robust loss functions -- it requires re-annotation.

3. **At this dataset scale (98K domains), LLM-generated keywords favor sparse methods -- but this ceiling is expected to break with scale.** On the constrained setup (single M4 Max, 78K training domains, E5-small 384-dim encoder), TF-IDF exploits the high mutual information between LLM-generated keywords and category labels. However, with production-scale data (millions of domains), larger embedding models (GTE-ModernBERT 768-dim, Matryoshka representations), and multi-GPU fine-tuning, dense architectures capture compositional semantics and handle distributional shift that bag-of-words cannot.

4. **Knowledge distillation pipeline is validated end-to-end and ready for scaling.** The 5.1x increase in teacher coverage (6.5K to 32.9K Opus-labeled domains) combined with ground truth correction produced 88% relative improvement in the MLP. The distillation architecture (temperature-scaled KL divergence from LLM soft labels into compact student networks) is the mechanism by which billion-parameter LLM knowledge transfers into production-deployable models -- this project validates the full machinery.

5. **Inference latency spans 500x across architectures -- the right tradeoff depends on deployment constraints.** TF-IDF: 0.02ms (sparse matrix multiply + SVM decision function). MLP: ~1ms (requires E5 forward pass for embedding generation). ModernBERT: ~10ms (full 150M-param forward pass). For RTB environments with strict latency budgets, pre-computation and model serving infrastructure (ONNX runtime, TensorRT, batched inference) make dense models viable at scale.

## The Dataset

Source: [Kaggle - Website IAB Categorization](https://www.kaggle.com/datasets/bpmtips/websiteiabcategorization) (98K domains)

| Split | Rows | Unique Domains |
|-------|------|----------------|
| Train | 78,357 | 77,588 |
| Val | 9,810 | 9,699 |
| Test | 9,794 | 9,699 |

27 IAB Tier-1 categories with 253x class imbalance (Shopping: 13,175 vs Sensitive Subjects: 52 in train). Class-weighted loss handles the imbalance across all models.

## Project Structure

```
notebooks/
  00_architecture_overview.ipynb          # System design reference (no code)
  v1-kaggle/                              # Pipeline on original noisy Kaggle labels
    01_data_preparation_eda.ipynb
    02_claude_teacher_labeling.ipynb
    03_embedding_generation.ipynb
    04_student_distillation_training.ipynb
    05_modernbert_finetune.ipynb          # 61.3% top-1 (noisy labels)
  v2-llm-corrected/                       # Pipeline on Sonnet-corrected labels
    01_data_correction_sonnet.ipynb       # 97K domains corrected (49.7% changed)
    02_eda.ipynb                          # Corrected dataset analysis
    03_embedding_generation.ipynb         # E5-small-v2 on enriched text
    04_student_distillation_training.ipynb # MLP: 84.9% top-1 (337K params)
    05_modernbert_finetune.ipynb          # ModernBERT: 150M params fine-tuning
    06_tfidf_baseline.ipynb               # TF-IDF: 91.6% top-1 (best model)

scripts/
  run_teacher_labeling.py                 # Batch Opus labeling (async, checkpointed)
  run_sonnet_correction.py                # Batch Sonnet correction (async, checkpointed)
  check_status.py                         # Monitor background processes

data/
  processed/                              # Original Kaggle data (train/val/test splits)
    embeddings/                           # E5-small-v2 embeddings (v1)
    teacher_checkpoints/                  # Opus soft label checkpoints (209 files)
    teacher_labels.parquet                # Consolidated Opus labels (40,696 domains)
  corrected/                              # Sonnet-corrected data
    sonnet_checkpoints/                   # Correction checkpoints (524 files)
    embeddings/                           # E5-small-v2 embeddings (v2)
    train.parquet, val.parquet, test.parquet

models/
  student_mlp_best.pt                     # v1 MLP (45.1% top-1)
  student_mlp_v2_best.pt                  # v2 MLP (84.9% top-1, 1.3 MB)
  tfidf_v2/                               # TF-IDF + LinearSVC (91.6% top-1, 20.6 MB)
  modernbert_v1_best/                     # v1 ModernBERT (61.3% top-1, 602 MB)
```

## Technical Stack

- **AWS Bedrock (async inference)** -- Sonnet 4 (label correction, 97K domains, 10 concurrent workers) and Opus 4.6 (soft teacher labels, 40K domains, 5 concurrent workers) via ThreadPoolExecutor with fault-tolerant checkpointing every 200 domains. Exponential backoff retry with 3 attempts per domain.
- **E5-small-v2** (33M params, ONNX-exportable) -- bi-encoder sentence-transformer producing L2-normalized 384-dim embeddings. Mean pooling over token representations with [CLS]-free architecture. Throughput: 1,184 domains/sec on Apple MPS backend.
- **TF-IDF vectorization** -- sublinear term frequency (1 + log(tf)), L2-normalized IDF weighting, vocabulary of 30K features (unigrams + bigrams, min_df=2, max_df=0.95). Produces sparse CSR matrices with 99.9% sparsity (~70 non-zero features per document).
- **LinearSVC** -- L2-regularized hinge loss (C=1.0), one-vs-rest decomposition over 27 classes, class-weighted sample importance for 253x imbalance handling. Platt scaling via CalibratedClassifierCV for probability outputs.
- **PyTorch MLP distillation** -- 3-layer network (384->512->256->27) with GELU activations, BatchNorm, Dropout 0.3. Composite loss: alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T)) + (1-alpha) * weighted_BCE. Temperature T=2.0, alpha=0.7. AdamW optimizer (lr=1e-3, weight_decay=1e-4), cosine annealing with warm restarts.
- **ModernBERT fine-tuning** -- 150M-param encoder (answerdotai/ModernBERT-base) with classification head. Discriminative learning rates (encoder: 2e-5, head: 1e-3), linear warmup over 10% of steps, gradient accumulation for effective batch size 64.

## How to Run

```bash
# Clone the repository
git clone https://github.com/nbatra/bert-finetune-iab-classification.git
cd bert-finetune-iab-classification

# Create environment and install dependencies
uv venv --python 3.13 .venv
uv pip install anthropic httpx torch sentence-transformers pandas numpy pyarrow scikit-learn transformers

# Set up AWS config (required for LLM labeling scripts)
cp config.json.template config.json
# Edit config.json: replace YOUR_ACCOUNT_ID with your AWS account ID

# Step 1: Run Sonnet data correction (async, ~4 hours, checkpoints every 200 domains)
.venv/bin/python scripts/run_sonnet_correction.py

# Step 2: Run Opus teacher labeling (async, ~4 hours, checkpoints every 200 domains)
.venv/bin/python scripts/run_teacher_labeling.py

# Step 3: Execute notebooks in order
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/v2-llm-corrected/03_embedding_generation.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/v2-llm-corrected/04_student_distillation_training.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/v2-llm-corrected/06_tfidf_baseline.ipynb
```

Full pipeline takes approximately 5 hours (dominated by async LLM API calls). Model training alone takes under 2 minutes for TF-IDF and MLP combined.

### Configuration

The `config.json` file (not committed -- see `config.json.template`) contains:
- AWS Bedrock model ARNs for Opus and Sonnet
- Worker counts and checkpoint intervals
- AWS region

Scripts read all model IDs from this config file. You need AWS Bedrock access with Claude models enabled.

### Sample Data

`data/sample/corrected_sample.csv` contains 135 sample rows (5 per category) from the corrected training set, showing:
- `domain_clean`: the website domain
- `tier1`: the Sonnet-corrected category
- `tier1_kaggle`: the original (often wrong) Kaggle label
- `keywords`: Sonnet-generated English keywords
- `text`: enriched text used for classification (domain | title | keywords)

### What's Included in the Repo

The repository includes all data and models needed to run notebooks end-to-end, except for files that exceed GitHub's 100MB limit:

| Included | Size | Description |
|----------|------|-------------|
| `data/corrected/{train,val,test}.parquet` | 55 MB | Full corrected dataset (78K/10K/10K rows) |
| `data/corrected/embeddings/` | 30 MB | Val/test embeddings + domain indices |
| `data/processed/teacher_labels.parquet` | 1 MB | Opus soft labels (40,696 domains) |
| `data/processed/{train,val,test}.parquet` | 34 MB | Original Kaggle splits |
| `models/student_mlp_v2_best.pt` | 2 MB | Trained MLP (84.9% top-1) |
| `models/tfidf_v2/` | 20 MB | Trained TF-IDF + LinearSVC (91.6% top-1) |

| Excluded (regenerate) | Size | How to Regenerate |
|-----------------------|------|-------------------|
| `data/corrected/embeddings/train_embeddings.npy` | 114 MB | Run notebook `v2/03_embedding_generation.ipynb` (~3 min) |
| `data/processed/embeddings/train_embeddings.npy` | 114 MB | Run notebook `v1/03_embedding_generation.ipynb` |
| ModernBERT checkpoints | 600 MB+ | Run notebook `v2/05_modernbert_finetune.ipynb` (~75 min) |
| LLM checkpoint JSONs | ~500 MB | Run scripts (requires AWS Bedrock access) |

To run the full pipeline after cloning, generate the train embeddings first:
```bash
# Generate train embeddings (required for notebook 04 -- MLP distillation)
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/v2-llm-corrected/03_embedding_generation.ipynb
```

Notebooks 01-02 (data correction, EDA) and 06 (TF-IDF) run directly on the included parquet files with no additional setup.

## Future Directions

| Approach | Architecture | Hypothesis |
|----------|-------------|------------|
| **GTE-ModernBERT embeddings** | Matryoshka representation learning (768-dim, nested dimensionality) | Higher intrinsic dimensionality preserves more discriminative variance than E5's 384-dim bottleneck |
| **LoRA fine-tuning on small LLMs** | Rank-16 adaptation on Gemma-3-4B or Phi-4-mini with IAB-conditioned generation | Direct parameter-efficient fine-tuning bypasses the two-stage pipeline; the student IS the teacher |
| **Dual-encoder retrieval** | Contrastive pre-training of domain and category descriptions in shared embedding space | Enables zero-shot generalization to unseen IAB categories without re-training |
| **PECOS / extreme multi-label** | Hierarchical label tree with balanced k-means partitioning (XR-Transformer architecture) | Required for Tier-2 expansion (700+ categories) where flat softmax becomes computationally intractable |
| **Multimodal fusion** | Late fusion of URL text features + rendered page screenshot (ViT-B/16 encoder) | Captures visual brand signals and layout patterns invisible to text-only pipelines |
| **Online distillation with curriculum** | Anti-curriculum ordering (hard examples first) with dynamic temperature scheduling | May improve MLP convergence on tail categories where soft labels carry most information |

---

## Author

Built by **Nipun Batra**

[![GitHub](https://img.shields.io/badge/GitHub-nbatra-181717?logo=github)](https://github.com/nbatra)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nipunbatra-0A66C2?logo=linkedin)](https://www.linkedin.com/in/nipunbatra/)

---

## License

This project is released for educational and portfolio purposes. The Kaggle dataset is provided under its own [terms of use](https://www.kaggle.com/datasets/bpmtips/websiteiabcategorization).

<details>
<summary>Keywords</summary>

IAB Content Taxonomy, website classification, domain categorization, knowledge distillation, LLM labeling, soft labels, teacher-student training, TF-IDF, LinearSVC, ModernBERT, fine-tuning, E5-small-v2, sentence embeddings, MLP classifier, multi-class classification, ad tech, contextual advertising, brand safety, real-time bidding, programmatic advertising, AWS Bedrock, Claude API, async batch processing, scikit-learn, PyTorch, Apple Silicon MPS, label correction, noisy labels, data quality, crowdsourced annotations, class imbalance, macro F1, top-k accuracy, sparse features, dense embeddings, model comparison, inference latency, production ML, NLP pipeline

</details>
