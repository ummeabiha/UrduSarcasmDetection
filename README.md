# Urdu Sarcasm Detection

A binary classifier for Urdu social-media posts: **sarcastic (1)** vs **non-sarcastic (0)**.
The project compares a classical TF-IDF model, two neural networks, a fine-tuned multilingual transformer and an ensemble of the three families,
with Urdu-specific preprocessing and features, an ablation study, statistical significance tests and an error analysis.

The notebook [`26K_7601_Urdu_Sarcasm_Detection.ipynb`](26K_7601_Urdu_Sarcasm_Detection.ipynb) is the full report: every number below comes from its outputs.

<img width="4177" height="2031" alt="architecture" src="https://github.com/user-attachments/assets/4f3d8ff4-0abe-46f1-ad30-fea2a7b87923" />

---

## Headline results

Held-out test set: 1,457 posts. Primary metric: **macro-F1** (the cleaned data is 59.4% / 40.6%, so accuracy alone would reward the majority class).

| Model | Validation macro-F1 | **Test macro-F1** | 95% CI (bootstrap) | Accuracy |
|---|---|---|---|---|
| **Ensemble** (final model, chosen on validation) | 0.8404 | **0.8329** | 0.8136 – 0.8527 | 0.8373 |
| **XLM-RoBERTa large** (best single model) | 0.8312 | **0.8365** | 0.8165 – 0.8554 | 0.8415 |
| BiGRU + attention | 0.8114 | 0.8070 | 0.7865 – 0.8283 | 0.8126 |
| Logistic Regression (TF-IDF + Urdu features) | 0.8083 | 0.8088 | 0.7886 – 0.8304 | 0.8161 |
| Linear SVM (TF-IDF + Urdu features) | 0.8083 | 0.8020 | 0.7803 – 0.8244 | 0.8126 |
| Multi-kernel CNN | 0.8056 | 0.7967 | 0.7733 – 0.8181 | 0.8071 |
| B2: mean FastText vector + LogReg (comparison only) | 0.7680 | 0.7594 | 0.7361 – 0.7801 | 0.7694 |

**Key findings**

- XLM-R and the Ensemble are the strongest models and are **not significantly different** from each other (exact McNemar test, p = 0.43).
- XLM-R is significantly better than BiGRU, Logistic Regression and Linear SVM (Holm-corrected McNemar p < 0.05).
- The classical model and the neural networks are statistically tied with each other.
- In the ablation, **no single preprocessing or feature choice changed the score significantly**: the model family matters far more than preprocessing.
- Validation and test scores agree within 0.009 for every model, so the selection process did not overfit the validation set.

### Per-class test results

| Model | Precision (0) | Recall (0) | F1 (0) | Precision (1) | Recall (1) | F1 (1) |
|---|---|---|---|---|---|---|
| Ensemble | 0.8794 | 0.8418 | 0.8602 | 0.7818 | 0.8308 | 0.8056 |
| XLM-R | 0.8757 | 0.8545 | 0.8650 | 0.7941 | 0.8223 | 0.8080 |
| BiGRU + attention | 0.8534 | 0.8268 | 0.8399 | 0.7573 | 0.7919 | 0.7742 |
| Logistic Regression | 0.8421 | 0.8499 | 0.8460 | 0.7770 | 0.7665 | 0.7717 |
| Linear SVM | 0.8192 | 0.8788 | 0.8479 | 0.8011 | 0.7157 | 0.7560 |
| CNN | 0.8176 | 0.8695 | 0.8428 | 0.7892 | 0.7157 | 0.7507 |

**Final model (Ensemble) confusion matrix on test**

| | Predicted non-sarcastic | Predicted sarcastic |
|---|---|---|
| **True non-sarcastic** (866) | 729 | 137 |
| **True sarcastic** (591) | 100 | 491 |

Confusion matrices for every model are in [`figures/confusion_matrices_test.png`](figures/confusion_matrices_test.png).

---

## Dataset

`urdu_sarcastic_dataset.xlsx` 20,004 rows, input column `urdu_text`, target `is_sarcastic`. Posts mix Urdu script, Roman Urdu and English, emojis, hashtags, mentions and URLs.

**Data-quality issues found in the EDA**

- 49 rows without text, and 6 stray "Unnamed" columns (two of them hold a few stray values); only the two required columns are kept.
- 4,689 exact duplicate rows, and 645 texts that appear with **both** labels. Identical posts disagree in 17.5% of pairs, which suggests roughly 10% label noise.
- The raw file looks balanced (50/50), but after de-duplication it is **59.4% non-sarcastic / 40.6% sarcastic**.
- Collection artifact: 554 posts contain `#sarcasm` or `#طنز`, and 99.1% of them are sarcastic. They are kept but measured separately.
- Sarcastic posts are longer on average (23.1 vs 16.1 words); 1–3-word posts are only 3.9% sarcastic.

## Pipeline

### 1. Cleaning and leakage-free split

1. Drop rows with missing text or a non-binary label.
2. Unicode NFKC normalisation and removal of invisible characters (zero-width joiners, BOM), so one post cannot exist as several strings.
3. Remove all 646 texts that occur with conflicting labels, then exact de-duplication → **14,567 unique posts**.
4. Stratified **80 / 10 / 10** split (seed 42): 11,653 train, 1,457 validation, 1,457 test.
5. Remove 452 training posts whose "core text" (no mentions, URLs, hashtags, punctuation, emojis or diacritics) matches a validation or test post → **11,201 training posts**.

The test set is read only once, after every setting, weight, threshold and the final-model choice have been fixed on validation.

### 2. Urdu-aware preprocessing

| Mode | What it does |
|---|---|
| `raw` | nothing beyond Unicode cleaning |
| `light` | URLs → `URL`, @mentions → `USER` |
| `full` | `light` + Arabic yeh/kaf → Urdu yeh/kaf, tatweel and diacritics removed, runs of 4+ identical characters cut to 3, `#` removed (word kept), Latin letters lower-cased |

`full` scored highest on validation and is used by the classical and neural models. XLM-R reads the cleaned text without normalisation.
Emojis, punctuation, repetition, hashtags and code-mixed words are **kept** and tested in the ablation instead of being removed blindly.

**Data augmentation:** each training post gets one typo-style copy (letter elongation, adjacent-letter swap or dropped letter), +10,426 copies.
It is used by a model family only if it improves validation macro-F1: it helped the neural networks, not the classical models, and XLM-R is trained on original posts only.

### 3. Feature engineering (bonus: Urdu-specific features)

1. **TF-IDF word 1–2-grams**, with a tokenizer that keeps each emoji, `!` and `?` as its own token.
2. **TF-IDF character 2–5-grams**, robust to spelling variation in Urdu and Roman Urdu.
3. **200 mined Urdu phrases**: the 100 most sarcastic and 100 most non-sarcastic bigrams/trigrams, selected from the training split with a smoothed log-odds ratio.
4. **11 sarcasm cue features**: emoji count, `!`, `?`/`؟`, ellipses, repeated-character runs, share of Latin words, quoted words, URL / mention / hashtag presence, log word count.
5. **Urdu sarcasm-marker lexicon** (extra experiment): mock praise (`واہ`, `شاباش`), mock agreement (`بالکل`), mock respect (`صاحب`, `جناب`), laughter, contrast, intensifiers, negation and rhetorical questions.

In total about 105k sparse features, all fitted on the training split only.

### 4. Models

| Model | Input | Architecture | Training and selection |
|---|---|---|---|
| **A. Logistic Regression** | TF-IDF + phrases + cues | linear, L2 (C = 1, balanced class weights) | C and class weight tuned on validation |
| **B. Linear SVM** | same as A | linear SVM (C = 0.1, balanced) + sigmoid calibration | C and class weight tuned on validation |
| **B2. Embedding baseline** | mean of FastText vectors | Logistic Regression (C = 0.1) | comparison only, never a candidate for the final model |
| **C. BiGRU + attention** | ≤ 60 tokens, FastText 100-d initialisation | BiGRU 2 × 128 → additive attention → dropout 0.3 → linear | Adam 1e-3, batch 64, early stopping (patience 3, max 15 epochs), 3 seeds averaged |
| **D. Multi-kernel CNN** | same as C | Conv1d with kernels 2, 3, 4, 5 (128 filters each) → max-pool → dropout 0.3 → linear | same as C |
| **E. XLM-RoBERTa large** | cleaned text, ≤ 128 subwords | pretrained encoder + classification head | lr 1e-5, warmup 10%, batch 16, ≤ 5 epochs, early stopping (patience 2), 3 seeds averaged |
| **Ensemble** | probabilities of A, C and E | weighted average: 0.3 × LogReg + 0.2 × BiGRU + 0.5 × XLM-R, threshold 0.46 | weights and threshold tuned on validation |

Why these choices for Urdu sarcasm:

- **Character n-grams and FastText subwords** absorb the many spellings of informal Urdu and Roman Urdu.
- **BiGRU with attention** can focus on the one or two words that turn a sentence sarcastic; the **CNN** detects short phrase patterns.
- **XLM-R** is pretrained on 100 languages including Urdu, so it brings knowledge of Urdu usage that 11k training posts cannot provide.
- **The ensemble** combines three model families that make different errors.

---

## Ablation study (validation)

Each variant changes one choice of the Logistic Regression setup and is compared with the base using an exact McNemar test.

| Variant | Validation macro-F1 | Δ vs base | McNemar p |
|---|---|---|---|
| Base (final setup, `full` mode) | 0.8083 | – | – |
| Preprocessing: `raw` | 0.8100 | +0.0017 | 0.79 |
| Preprocessing: `light` | 0.8084 | +0.0001 | 1.00 |
| Emojis removed | 0.8028 | −0.0055 | 0.38 |
| Hashtags removed | 0.8038 | −0.0045 | 0.26 |
| Punctuation removed | 0.8127 | +0.0044 | 0.29 |
| Repetition collapsed | 0.8095 | +0.0012 | 0.83 |
| All Latin-script (code-mixed) tokens removed | 0.8097 | +0.0014 | 0.69 |
| Word n-grams only | 0.8072 | −0.0011 | 0.92 |
| Character n-grams only | 0.8039 | −0.0044 | 0.41 |
| No phrase or cue features | 0.8098 | +0.0015 | 1.00 |
| + Urdu marker lexicon | 0.8105 | +0.0022 | 0.61 |
| Augmentation on | 0.8090 | +0.0007 | 1.00 |

None of the 14 variants, the 11 single-cue removals or the 9 phrase-mining settings is significant (every p ≥ 0.25, every Δ < 0.006).
The full tables are in `results/ablation_validation.csv`, `results/ablation_cue_by_cue.csv` and `results/phrase_mining_sweep.csv`.

## Further evaluation

- **Statistical tests:** pairwise exact McNemar tests for all 21 model pairs, Holm-corrected (`results/pairwise_mcnemar_test.csv`).
- **Calibration:** Ensemble Brier score 0.117, ECE 0.027 (base-rate reference 0.241).
- **Robustness:** one typo changes only 1.4–3.8% of predictions.
- **Test slices:** short posts (1–5 words) are the hardest (Ensemble macro-F1 0.660). On posts without the `#sarcasm`/`#طنز` tag, the Ensemble reaches 0.822 and XLM-R 0.826.
- **Sanity checks:** majority-class baseline 0.373 and shuffled-label model 0.503 macro-F1, which rules out leakage through the features.

## Error analysis

The final model makes 237 test errors (137 false positives, 100 false negatives). All errors are tagged by rules, and 12 misclassified posts are read and explained by hand.
The main error sources:

- **Doubtful labels:** for example, a post ending in `#Sarcasm` that is labelled non-sarcastic.
- **Political mockery and idioms that look sarcastic:** `بلی کے گلے میں گھنٹی باندھنا`, `کانپیں ٹانگ جاتی ہیں`, and abuse followed by laughing emojis.
- **Missing context:** short praise or thanks (`عمدہ👌`, `شکریہ جویریہ 😊😊`) that is only sarcastic as a reply to another post.
- **Spelling variation:** tatweel-stretched words (`یــــــقین`) become rare character patterns.
- Misclassified posts are significantly longer than correct ones (Mann-Whitney p = 0.012).

## Limitations

- About 10% label noise in the raw data, and some confident "errors" are labelling errors.
- Conflicting-label posts were removed, so the test set under-represents the most ambiguous cases.
- The `#sarcasm`/`#طنز` tags are a collection artifact that partly reveals the label.
- Many posts are replies whose sarcasm depends on a conversation that is not in the dataset.
- A single train/validation/test split: differences below about 0.02 macro-F1 are within noise.

---

## Repository structure

```
├── 26K_7601_Urdu_Sarcasm_Detection.ipynb   notebook with all code, outputs and discussion (the report)
├── architecture.png                        pipeline diagram
├── results/                                predictions, metrics, ablation, significance and slice tables (CSV/JSON)
├── figures/                                all plots (EDA, training curves, confusion matrices, calibration)
├── config/                                 experiment configuration, ensemble weights, package versions
├── classical/                              fitted vectorizers + Logistic Regression and Linear SVM (joblib)
└── neural/                                 BiGRU and CNN weights, vocabulary, config
```

The XLM-R weights (about 3.4 GB) are too large for GitHub and are shared separately (see below).

## How to reproduce

1. Open the notebook in **Google Colab**.
2. Optional: to skip retraining, copy this repository's `classical/`, `neural/`, `results/` and `config/` folders, and the XLM-R weights from
   [Google Drive](https://drive.google.com/drive/folders/1po-lXi723uMiPew3uEl62Au5VOwM9zQ5?usp=drive_link) (into `transformer/`), under `MyDrive/Urdu_Sarcasm_Detection/`.
3. Keep `RESUME = True` and choose **Runtime → Run all**. The notebook mounts Drive and installs its extra packages itself.

- With the saved models, everything runs on a CPU in a few minutes.
- Training from scratch: the classical and neural models train on a CPU, but **fine-tuning XLM-R needs a GPU**. Without a GPU and without saved runs, XLM-R is skipped.
- All randomness is seeded (seed 42; seeds 42–44 for the averaged runs).

**Main package versions:** Python 3.13, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1, PyTorch 2.11.0, Transformers 5.18.0, gensim 4.4.0, matplotlib 3.10.0.
The exact versions are recorded in `config/package_versions.json`.

### Predicting new posts

After running the notebook, `predict_sarcasm` reloads the saved models and classifies raw Urdu text:

```python
predict_sarcasm(['واہ کیا بات ہے، بجلی پھر چلی گئی 👏👏 شاباش حکومت'])                        # final model (Ensemble)
predict_sarcasm(['واہ کیا بات ہے، بجلی پھر چلی گئی 👏👏 شاباش حکومت'], model_name='XLM-R')    # a single model
```

