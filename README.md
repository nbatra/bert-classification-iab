# IAB Content Taxonomy Classification via LLM Knowledge Distillation

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

**Keywords:** `IAB Content Taxonomy` `Knowledge Distillation` `Domain Classification` `LLM Labels` `TF-IDF` `LinearSVC` `E5 Embeddings` `Sentence Transformers` `ModernBERT` `Soft Labels` `KL Divergence` `Label Correction` `AdTech` `Programmatic Advertising` `Multi-class Classification` `Text Classification` `NLP` `Real-time Inference` `Production ML` `AWS Bedrock`

> Developed and trained entirely on a MacBook M4 Max (64GB RAM, MPS acceleration). The full pipeline -- from LLM data correction (97K domains) through embedding generation and model training -- completes in under 5 hours. The LLM labeling runs asynchronously via AWS Bedrock while all model training (TF-IDF, MLP distillation, ModernBERT fine-tuning) fits within Apple Silicon's memory budget. The compute constraint mirrors production economics: the best model turns out to be the simplest one.

## What This Project Is

This is a comparative study of **website domain classification** into IAB Content Taxonomy categories -- the core ML problem behind real-time ad targeting, brand safety, and contextual advertising. The central question: **when an LLM pre-classifies your features, does model complexity still matter?**

We use the [**Kaggle Website IAB Categorization**](https://www.kaggle.com/datasets/bpmtips/websiteiabcategorization) dataset -- 98K website domains with noisy crowd-sourced labels across 27 IAB Tier-1 categories. The critical finding: **49.7% of the original labels are wrong**, and fixing them with an LLM matters more than any model architecture choice.

The project implements a two-LLM pipeline (Sonnet for label correction + keyword generation, Opus for soft teacher labels) and then compares four downstream classifiers that represent distinct approaches to exploiting those labels:

1. **TF-IDF + LinearSVC** -- sparse bag-of-words on LLM-generated keywords (the winner)
2. **MLP + KL Distillation** -- dense embeddings learning from soft teacher distributions
3. **ModernBERT Fine-tuning** -- full 150M-param encoder trained end-to-end
4. **Logistic Regression / SGD** -- linear baselines on the same TF-IDF features

## The Core Insight: LLM Keywords Are Pre-Classified Features

The surprising result of this project is that the simplest model wins. TF-IDF + LinearSVC (91.6%) beats the distilled MLP (84.9%) by 6.7 percentage points, and does so 50x faster at inference.

Why? When Sonnet generates keywords like "online store, electronics, gadgets" for an e-commerce domain, those keywords ARE the classification. TF-IDF converts them directly into high-IDF discriminative features -- the word "automotive" maps nearly 1:1 to the "Autos & Vehicles" category. Dense embeddings (E5-small, 384-dim) actually lose information by compressing these explicit lexical signals into a continuous space where "automotive" and "car dealership" merge.

This does NOT mean TF-IDF is generally better than neural approaches. It means that **when the input text contains LLM-generated category-aligned keywords, bag-of-words models exploit those signals more directly than dense embeddings**. The keywords were generated specifically to describe category membership -- they are effectively pre-classified features expressed in natural language.

## The Two-LLM Architecture

A two-model approach where each LLM handles a distinct task:

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
| TF-IDF + Logistic Regression | 90.4% | 99.4% | -- | 0.878 | ~30K features | 0.02 ms |
| MLP (E5 + KL distillation) | 84.9% | 98.3% | -- | 0.814 | 337K | ~1 ms |
| ModernBERT (corrected labels) | (in progress) | -- | -- | -- | 150M | ~10 ms |

### v1 Baseline (noisy Kaggle labels, for comparison)

| Model | Val Top-1 | Val Top-3 | Params |
|-------|-----------|-----------|--------|
| v1 ModernBERT | 61.3% | 80.0% | 150M |
| v1 MLP (distilled) | 45.1% | 68.0% | 135K |
| Random baseline | 3.7% | 11.1% | -- |

### The Impact of Label Correction (v1 to v2)

| Factor | v1 (noisy labels) | v2 (corrected) | Change |
|--------|-------------------|----------------|--------|
| MLP Top-1 accuracy | 45.1% | 84.9% | +88% relative |
| MLP Top-3 accuracy | 68.0% | 98.3% | +45% relative |
| Teacher coverage | 8.4% (6.5K domains) | 42.4% (32.9K domains) | 5.1x |
| Label quality | 50.9% Kaggle-Opus agreement | 81.7% Sonnet-Opus agreement | +60% |

The v2 MLP (337K params, 1.3 MB) outperforms the v1 ModernBERT (150M params, 602 MB) by 23.6 percentage points -- proving that **label quality dominates model capacity** for classification tasks.

### Why These Results Make Sense

**TF-IDF wins because the features are pre-classified.** Sonnet's keywords ("online store, electronics, gadgets") are essentially the category expressed in natural language. TF-IDF treats "automotive" as a near-perfect indicator feature with high IDF weight. The LinearSVC's margin maximization on this 30K-dimensional sparse space (99.9% sparsity) is the optimal classifier for such features.

**The MLP trails by 6.7pp despite using Opus soft labels.** E5-small compresses the keywords into 384 dense dimensions, losing lexical precision. "Automotive" and "car dealership" become similar vectors -- useful for generalization, but harmful when the original tokens were already discriminative enough.

**Hyperparameters barely matter for TF-IDF.** Sensitivity analysis shows only 1.4pp spread across all configurations (20K-50K features, unigrams through trigrams). The signal is concentrated in a small set of high-value keyword features.

**Val/Test consistency is excellent.** 91.6% val vs 91.7% test (0.1pp difference) -- no overfitting.

### Per-Category Performance (TF-IDF + LinearSVC)

Strong performers (F1 > 0.95): Adult (0.963), Travel (0.962), Real Estate (0.958), Jobs & Education (0.955), News (0.954), Autos & Vehicles (0.953), Finance (0.950), Health (0.950)

Weak performers: Sensitive Subjects (0.000 -- only 3 val samples), Hobbies & Leisure (0.812), Online Communities (0.833), Science (0.854)

## Key Findings

1. **49.7% of Kaggle labels were wrong.** Sonnet and Opus independently agree on corrections for 35.6% of domains where Kaggle had the wrong label. Shopping was the most undercounted category (+104% after correction) -- e-commerce sites had been scattered across Home & Garden, Business & Industrial, etc. Sensitive Subjects was 99% wrong (1.0% agreement).

2. **Label quality > model size.** A 337K-param MLP with clean labels beats a 150M-param transformer with noisy labels by 23.6 percentage points. The TF-IDF model with clean labels achieves 91.6% with no neural network at all.

3. **LLM keywords make TF-IDF competitive with deep learning.** When the upstream LLM has already done the semantic heavy lifting (extracting category-relevant keywords), a simple linear model on bag-of-words features outperforms embedding-based approaches. The model choice matters less than the feature generation.

4. **5.1x more teacher labels + corrected ground truth = 88% relative improvement.** Going from 6.5K to 32.9K Opus-labeled domains AND fixing the ground truth produced the v1-to-v2 accuracy jump. In v1, we proved that more teacher labels without fixing ground truth did not help -- the noisy label ceiling was the bottleneck.

5. **TF-IDF is 50x faster than embedding-based inference.** At 0.02ms per domain (vectorization + classification), TF-IDF is the fastest production option. The MLP requires E5 embedding computation (~1ms), and ModernBERT requires a full forward pass (~10ms).

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

- **AWS Bedrock** -- Sonnet 4 (label correction, 97K domains) and Opus 4.6 (soft teacher labels, 40K domains) via async batch processing with automatic checkpointing
- **E5-small-v2** (33M params) -- sentence-transformer encoding domain text into 384-dim embeddings at 1,184 domains/sec on MPS
- **scikit-learn** -- TF-IDF vectorization (30K features, unigrams + bigrams, sublinear TF) and LinearSVC with calibration
- **PyTorch** (MPS backend) -- MLP distillation training (KL divergence + weighted BCE) and ModernBERT fine-tuning
- **Knowledge distillation** -- alpha=0.7 KL divergence on Opus soft labels + alpha=0.3 weighted BCE on corrected hard labels

## How to Run

```bash
# Clone the repository
git clone https://github.com/nbatra/iab-domain-classification.git
cd iab-domain-classification

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

| Approach | Core Idea | Expected Impact |
|----------|-----------|-----------------|
| **GTE-ModernBERT embeddings** | Drop-in encoder upgrade (768-dim, Matryoshka) | +2-5% for MLP approach |
| **Small LLM fine-tuning** | LoRA on Gemma-3-4B or Phi-4-mini | Higher quality ceiling than distillation |
| **Two-Tower retrieval** | Encode domains and categories in same space | Zero-shot for new categories |
| **PECOS** | Hierarchical label partitioning | Required for Tier-2 expansion (700 classes) |
| **Multimodal** | URL + page screenshot | Handles domains with minimal text metadata |

---

## Author

Built by **Nipun Batra**

[![GitHub](https://img.shields.io/badge/GitHub-nbatra-181717?logo=github)](https://github.com/nbatra)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nipunbatra-0A66C2?logo=linkedin)](https://www.linkedin.com/in/nipunbatra/)

---

## License

This project is released for educational and portfolio purposes. The Kaggle dataset is provided under its own [terms of use](https://www.kaggle.com/datasets/bpmtips/websiteiabcategorization).
