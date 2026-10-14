# Urdu Sarcasm Detection
Binary classifier for Urdu social-media posts: sarcastic (1) vs non-sarcastic (0).
The notebook is the full report: EDA, preprocessing, features, models, ablation, evaluation and error analysis.

## Results (held-out test set, 1,457 posts, macro-F1)

| Model | Test macro-F1 | Accuracy |
|---|---|---|
| XLM-RoBERTa large (best single model) | 0.8365 | 0.8415 |
| Ensemble (final model, chosen on validation) | 0.8329 | 0.8373 |
| Logistic Regression (TF-IDF + Urdu features) | 0.8088 | 0.8161 |
| BiGRU + attention | 0.8070 | 0.8126 |
| Linear SVM | 0.8020 | 0.8126 |
| CNN | 0.7967 | 0.8071 |

## Repository contents
- `Urdu_Sarcasm_Detection.ipynb` – notebook with all outputs
- `results/` – predictions, metrics, ablation and statistical-test tables (CSV)
- `figures/` – all plots
- `config/` – experiment configuration, ensemble weights, package versions
- `classical/`, `neural/` – saved classical and neural models

## Reproducing
1. Open the notebook in Google Colab.
2. XLM-R weights (too large for GitHub): [Google Drive folder](https://drive.google.com/drive/folders/1po-lXi723uMiPew3uEl62Au5VOwM9zQ5?usp=sharing).
   Copy them into `MyDrive/Urdu_Sarcasm_Detection/transformer/`.
3. Keep `RESUME = True` and run all cells. Training XLM-R from scratch needs a GPU.
