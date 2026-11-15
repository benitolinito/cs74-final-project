# CS74 - Algorithmic Equity in Moderating Toxic Comments

**Ben Folk & Ben Lin**

## Project overview

**Research question:** Can automated toxicity moderation detect harmful language without disproportionately flagging comments that mention particular identities?

We trained binary classifiers to label comments as **toxic (1)** or **non-toxic (0)**. We compared toxicity-detection performance across models and examined whether non-toxic comments mentioning particular identities were incorrectly flagged at different rates.

Our main analysis is in [`CS74_Project_Final_Version.ipynb`](CS74_Project_Final_Version.ipynb).

**Colab notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1kZRUNdp4kbKtHRcBp_zvL9KvJwLOsXTk?usp=sharing)

**Presentation slides:** [Google Slides](https://docs.google.com/presentation/d/1pitYnRS2cjvZANsdkaCI5XGWoHZWspcXSRKSz4958nM/edit?usp=sharing)

## Dataset and preprocessing

We used the **Jigsaw Unintended Bias in Toxicity Classification (Civil Comments)** dataset, available from [Hugging Face](https://huggingface.co/datasets/shuttie/jigsaw-unintended-bias). The notebook downloads `data/train.csv.gz` directly, so the full dataset is not stored in this repository.

We removed missing comment texts and toxicity scores, dropped duplicate comment texts, and defined a binary toxicity label using `target >= 0.5`. We made stratified **70% training / 20% validation / 10% test** splits with `random_state=42`.

Identity-mention groups are defined using annotation scores of at least `0.5`. These annotations describe **identities mentioned in comments**, not the identities of their authors. We use them for evaluation rather than as predictive features.

## Methods

We compared five handcrafted features (word count, uppercase-letter ratio, exclamation count, question-mark count, and profanity count) with TF-IDF text features (unigrams and bigrams, up to 40,000 features). The initial model experiments used stratified samples of **100,000 training** and **30,000 validation** comments. KNN used a smaller **20,000-comment training subset** for computational reasons.

Classification models evaluated:
- Logistic Regression with handcrafted features and with TF-IDF
- Linear SVM and Perceptron with TF-IDF
- Multinomial Naive Bayes with TF-IDF
- Gaussian Naive Bayes and KNN with handcrafted features

Baseline: We also evaluated a majority-class classifier that predicts every comment as non-toxic to provide a reference point for performance.

We tuned the Linear SVM regularization parameter using validation binary F1 and examined how the Logistic Regression classification threshold affected precision, recall, F1, and false-positive rate. We then compared four TF-IDF classifiers—Logistic Regression, Linear SVM, Perceptron, and Multinomial Naive Bayes—at approximately matched overall false-positive rates, with thresholds chosen on validation data.

## Evaluation and findings

We report **binary F1** for the toxic class, **macro F1**, precision, recall, accuracy, and ROC-AUC. To study potential moderation disparities, we calculate false-positive rates (FPRs) for identity-mention groups, together with sample sizes and 95% Wilson confidence intervals. We also inspect misclassified comments, Logistic Regression feature weights, and overlap between the models' test-set errors.

In the held-out test evaluation, TF-IDF Logistic Regression and Linear SVM achieved similar binary F1 scores (approximately **0.566** and **0.568**, respectively). Although their overall false-positive rates were also similar, some identity-mention groups had substantially different false-positive rates across the two models. The Perceptron and Multinomial Naive Bayes reduced FPR for some groups but had lower toxic-class F1, illustrating a trade-off between false positives and missed toxic comments.

These findings are **descriptive rather than causal**. Identity categories can overlap, annotations are available for only part of the data, and the toxicity label depends on a 0.5 cutoff. Our primary fairness analysis focuses on false positives rather than subgroup false negatives. Further interpretation and future work are discussed at the end of the notebook.

## Reproducing the analysis

1. Open `CS74_Project_Final_Version.ipynb` in **Google Colab** with a Python 3 runtime.
2. Use **Runtime → Restart session and run all** to execute the cells in order. Internet access is needed to download the dataset (about 286 MB) from Hugging Face.
3. The first code cell installs `huggingface_hub`, `scikit-learn`, and `statsmodels`. The notebook also uses `pandas`, `numpy`, and `matplotlib`, which are typically available in Colab.
4. The notebook generates the model comparisons, subgroup fairness tables, plots, and test-set diagnostics.

A full run took about 3 minutes on a laptop; expect somewhat longer in Colab, including the dataset download.

All parameter choices and thresholds are documented in the notebook. We use validation data to choose them and reserve the test split for final evaluation. Results may vary slightly with package versions and the execution environment: re-running with a different scikit-learn version changed overall F1 and ROC-AUC by less than 0.01, while subgroup FPRs for small groups (e.g., transgender, with 196 non-toxic test comments) shifted by up to a few percentage points. The main conclusions did not change.

## References

- **Dataset:** Jigsaw. (2019). *Jigsaw Unintended Bias in Toxicity Classification*. Kaggle. https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification (the notebook downloads a copy hosted on Hugging Face: https://huggingface.co/datasets/shuttie/jigsaw-unintended-bias)
- **Related research:** Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification*. WWW '19 Companion. https://doi.org/10.1145/3308560.3317593
- **scikit-learn:** Scikit-learn Developers. *Scikit-learn Documentation*. https://scikit-learn.org/stable/
- **statsmodels:** Statsmodels Developers. *Statsmodels Documentation*. https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
