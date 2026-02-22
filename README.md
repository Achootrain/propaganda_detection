# Propaganda Detection and Classification in News Articles

A propaganda detection system for news articles based on **SemEval-2020 Task 11**, built with **PyTorch** and **Hugging Face Transformers**. The system performs two tasks: identifying propaganda spans in text and classifying them into 14 technique types.

---

## Framework & Libraries

| Framework | Role |
|-----------|------|
| **PyTorch** | Deep learning framework for model training and inference |
| **Hugging Face Transformers** | Pre-trained RoBERTa-large backbone and tokenizers |
| **spaCy** (`en_core_web_sm`) | Linguistic feature extraction (POS, NER, dependency parsing) |
| **seqeval** | Sequence labeling evaluation (precision, recall, F1) |
| **scikit-learn** | Metrics and class weight computation |
| **Flask** | Interactive web demo for end-to-end inference |

---

## Training Techniques

### Task 1 — Span Identification

Architecture: **RoBERTa-large + BiLSTM + CRF** (`BertLstmCrf`)

| Technique | Description |
|-----------|-------------|
| **ScalarMix** | Learns a weighted combination of all RoBERTa hidden layers instead of using only the last layer |
| **BiLSTM** | 2-layer Bidirectional LSTM (hidden size 600) captures sequential context on top of transformer representations |
| **CRF Decoding** | Linear-chain Conditional Random Field enforces valid BIO tag transitions during inference |
| **Auxiliary Linguistic Features** | POS, NER, and dependency label embeddings are concatenated with token representations |
| **Multitask Learning** | Jointly trains with two auxiliary objectives: sentence-level propaganda detection and multi-label technique classification (weighted at 0.5) |
| **Encoder Freezing** | Freezes the RoBERTa encoder for the first 2 epochs, then fine-tunes end-to-end |
| **BIO Tagging** | Frames span detection as a token-level sequence labeling problem with Begin/Inside/Outside tags |
| **Early Stopping** | Monitors dev set performance with patience of 5 epochs |
| **AdamW + Warmup** | AdamW optimizer (lr=2e-5, weight decay=0.01) with 500 warmup steps |

### Task 2 — Technique Classification

Architecture: **Fine-tuned RoBERTa-large** (`CustomRobertaForSequenceClassification`)

| Technique | Description |
|-----------|-------------|
| **Enriched Classification Head** | Combines `[CLS]` token, mean-pooled span embeddings, span length, and label prior matchings (default head variant) |
| **Inverse-Frequency Class Weighting** | Cross-entropy loss weighted by inverse class frequency to handle severe label imbalance |
| **Transfer Learning from SI** | Optionally initializes the RoBERTa backbone from the Span Identification checkpoint |
| **Post-Processing Heuristics** | Rule-based corrections: repetition detection, slogans boosting, train-span boosting, sub/super-span label consistency |
| **Multi-Seed Ensemble** | Trains with multiple random seeds (22, 42, 69, 123, 1024) and aggregates predictions via majority voting |
| **Early Stopping** | Monitors dev set performance with patience of 5 epochs |

---

## Detected Propaganda Techniques (14 Classes)

Appeal_to_Authority, Appeal_to_fear-prejudice, Bandwagon/Reductio_ad_hitlerum, Black-and-White_Fallacy, Causal_Oversimplification, Doubt, Exaggeration/Minimisation, Flag-Waving, Loaded_Language, Name_Calling/Labeling, Repetition, Slogans, Thought-terminating_Cliches, Whataboutism/Straw_Men/Red_Herring

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Train & evaluate Span Identification
python -m span_identification

# Train & evaluate Technique Classification
python -m technique_classification

# Run the web demo
cd demo && python app.py    # http://0.0.0.0:8000
```
