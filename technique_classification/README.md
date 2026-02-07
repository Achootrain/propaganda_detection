# Technique Classification Model

This directory contains the implementation for **Task 2 (Technique Classification)**. The model is designed to classify the propaganda technique used in a specific span of text, given the span and its context.

## Overview

The core of the solution is a fine-tuned **RoBERTa** model (`roberta-large` by default) with a **custom classification head** that incorporates additional features and embeddings beyond the standard `[CLS]` token.

## Model Architecture

The model is implemented in `CustomRobertaForSequenceClassification` and supports several architectural variants based on configuration flags.

### Base Encoder
- **Backbone**: `RobertaModel` (Hugging Face Transformers).
- **Input**: Tokenized text sequences representing the target span and its surrounding context.

### Custom Classification Heads
The model dynamically selects a classification head based on the enabled features:

1.  **Standard Head** (`RobertaClassificationHead`):
    - Uses only the `[CLS]` token embedding.
    - Standard dense -> tanh -> dropout -> projection.

2.  **Length-Augmented Head** (`RobertaClassificationHeadLength`):
    - Concatenates the `[CLS]` embedding with a **scalar length feature** (number of words in the span).

3.  **Matchings-Augmented Head** (`RobertaClassificationHeadMatchings`):
    - Concatenates the `[CLS]` embedding with a **14-dimensional matchings vector**.
    - This vector indicates which labels the current span text (stemmed) was associated with in the training set.

4.  **Joined Embeddings Head** (`RobertaClassificationHeadJoined`):
    - Concatenates the `[CLS]` embedding with a **pooled embedding** of the span tokens.
    - The pooled embedding is the mean (or masked sum) of the embeddings for the tokens corresponding to the span, providing a more direct representation of the span's content than the `[CLS]` token alone.

5.  **Joined + Length Head** (`RobertaClassificationHeadJoinedLength`):
    - Combines all valid signals: `[CLS]` embedding + Scalar Length + Pooled Span Embeddings.

### Input Features
The `InputFeatures` pipeline generates the following for each example:
- **`input_ids`, `attention_mask`**: Standard tokenizer outputs.
- **`length_feat`**: The number of words in the span.
- **`matchings`**: A one-hot-like vector (normalized) representing the prior probability of this text span belonging to specific classes based on training data statistics.

## Post-Processing (Inference)

The pipeline uses a sophisticated post-processing step (`postprocess_predictions_local`) to refine predictions, mimicking the logic from the original winning solution paper:

1.  **Repetition Heuristic**: specific logic to detect the "Repetition" technique based on the number of times a normalized span appears in the article.
2.  **Train-Span Boosting**: If a span text was seen in the training data with a specific label, the logit for that label is boosted (+0.5).
3.  **Slogans Heuristic**: Spans starting with `#` are heavily boosted towards the "Slogans" label.
4.  **Multi-label Handling**: If the exact same span coordinates appear multiple times (indicating a multi-label case), the system tries to force different predictions for each instance.
5.  **Consistency Constraints**: Uses sub-span/super-span co-occurrence statistics (`insides` map) from the training data to resolve conflicts between nested spans (e.g., if a span is "Name Calling", its sub-span is unlikely to be "Slogans" unless that pair was seen in training).

## Training

- **Loss Function**: Cross-Entropy Loss with **class weights** to handle the heavy class imbalance.
- **Optimizer**: AdamW.
- **Scheduler**: Linear schedule with warmup.
- **Early Stopping**: Monitors specific metrics (Macro F1) on the development set.
- **Initialization**: Can initialize from a standard pre-trained model or a custom checkpoint (supporting Transfer Learning).

## Usage

The script `technique_classification.py` is an end-to-end pipeline.

### Defaults
- **Model**: `roberta-large`
- **Max Sequence Length**: 256
- **Batch Size**: 8 (train), 16 (eval)
- **Epochs**: 15
- **Learning Rate**: 2e-5

### Running
Ensure your data is in the expected directory structure (or modify `DATA_ROOT` env var).

```bash
python -m technique_classification
```

### Directory Structure Assumption
```
datasets/
    train-articles/
    dev-articles/
    test-articles/
    train-task2-TC.labels
    dev-task2-TC.labels
    test-task2-TC-template.labels
```
