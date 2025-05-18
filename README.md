# Sentiment-Decoder-Pro
Fine-tuned RoBERTa for sentiment analysis using Hugging Face Transformers. Compared standard full-model tuning vs. efficient LoRA-based tuning. Achieved strong performance with reduced resources. Includes training, metrics, and visualization.
#  Sentiment Analysis with RoBERTa: Full vs. LoRA Fine-Tuning

This project demonstrates sentiment classification using the **RoBERTa** transformer model. It compares standard fine-tuning with **Low-Rank Adaptation (LoRA)**, highlighting how LoRA reduces training cost without compromising performance.

---

##  Background

###  What is RoBERTa?

RoBERTa (Robustly Optimized BERT Approach) is a transformer-based language model that improves on BERT by:
- Removing Next Sentence Prediction (NSP)
- Using larger mini-batches and more training data
- Training longer

It performs exceptionally well on various NLP tasks, especially **sentiment analysis**, due to its deep contextual understanding of text.

###  What is LoRA?

**Low-Rank Adaptation (LoRA)** is a parameter-efficient fine-tuning technique where only small matrices are trained, while the rest of the pretrained model remains frozen.

**Benefits of LoRA**:
- Reduces GPU memory usage
- Trains faster
- Prevents overfitting on small datasets

---

##  Project Objectives

1. Fine-tune **RoBERTa** on a sentiment dataset
2. Compare two approaches:
   - Full fine-tuning (Part 1)
   - LoRA fine-tuning (Part 2)
3. Evaluate performance using metrics like accuracy and F1 score
4. Visualize confusion matrix

---

##  Project Structure

- `assignment_3_part_1.ipynb`: Full fine-tuning of RoBERTa for 2 epochs
- `assignment_3_part_2.ipynb`: Fine-tuning using LoRA for better resource efficiency

---

## Technical Details

- **Language Model**: `roberta-base`
- **Libraries**:
  - `transformers`, `datasets`, `peft` (for LoRA)
  - `sklearn` for evaluation
  - `seaborn`, `matplotlib` for visualization

---

##  Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1 Score**
- **Confusion Matrix Visualization**

LoRA maintained high performance while using fewer trainable parameters.

---

