# Week 5 — Text Generation using RNN, LSTM and GRU

**Intern:** Soham Sharma  
**LMS ID:** CT_CSI_DS_1031  

---

## Objective

Build and compare three sequence models: Vanilla RNN, LSTM and GRU, on a next-word prediction task, and evaluate their ability to learn grammar, contextual dependencies and sentence flow from a text corpus.

---

## Topics Covered

- Tokenization and N-gram Sequence Creation
- Padding and Sequence Alignment
- Train / Validation Split
- Vanilla RNN Architecture and Vanishing Gradient Problem
- LSTM : Input, Forget and Output Gates
- GRU : Update Gate and Reset Gate
- Next-Word Prediction
- Training Loss vs Validation Loss
- Overfitting in Small Corpus Settings
- Text Generation with Greedy Decoding

---

## Key Results

| Model | Parameters | Train Loss (final) | Val Loss (final) |
|---|---|---|---|
| Vanilla RNN | 39,758 | low | highest |
| GRU | 89,550 | low | mid |
| LSTM | 113,870 | low | lowest |

**Vocabulary size:** 78
**Total sequences:** 106 (90 train / 16 val)
**Max sequence length:** 12
**Epochs:** 200 | **Embedding dim:** 64 | **Hidden units:** 128

**Generated text (seed: "neural networks are used for")**

```
rnn : neural networks are used for the data has some order like text little train one
lstm: neural networks are used for when the data has some order like text text text
gru : neural networks are used for the data has some order like like text text text
```

---

## Note

All three models overfit heavily given the small corpus size (90 training samples, 200 epochs), training loss drops close to zero while validation loss climbs back up after a certain point. The repeated word output ("text text text") in LSTM and GRU generations is a direct consequence of this overfitting. A larger corpus or an EarlyStopping callback monitoring val_loss would reduce this behaviour and produce more coherent generated text.

LSTM achieved the lowest validation loss overall, confirming its advantage in capturing longer contextual dependencies. GRU came close with ~21% fewer parameters, consistent with the general finding that GRU is a more efficient alternative to LSTM on smaller datasets.

---

## Stack

`tensorflow` `numpy` `matplotlib`
