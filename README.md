# HMS – Harmful Brain Activity Classification: scikit-learn baseline

Baseline for the Kaggle competition
[HMS – Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification).
Predictions are probability distributions over six EEG patterns, scored with the **KL divergence** to the
expert votes (lower is better).

* `hms_sklearn_baseline.ipynb` – the notebook to run on Kaggle.
* `hms_sklearn_baseline.py` – the same code as a script for your own computer.

## Quick for experiments, full for submissions

Set `MODE` in the second cell of the notebook:

| Mode | Use it for | Training data | Run time | Private KL | Public KL |
|---|---|---|---|---|---|
| `MODE = "quick"` | experiments | random 3,000 spectrograms | ~5 min | 0.9048* | 0.8186* |
| `MODE = "full"` | final submission | all spectrograms | ~10 min | to be measured | to be measured |

\* measured with quick mode and cross-validation off (`RUN_CV = False`).

## Run on Kaggle

1. Open the [competition page](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
   and accept the rules.
2. **Code → New Notebook**, then **File → Import Notebook** and upload `hms_sklearn_baseline.ipynb`.
3. Under **Session options**: **Accelerator = None**, **Internet = off**.
4. Set `MODE`, then **Save Version → Save & Run All (Commit)**. The log shows the cross-validation scores.
5. To submit: open the saved version, **Output → Submit to Competition**. The score appears on the
   **Submissions** page after 10–20 minutes.

## Run locally

Needs ~30 GB of disk for the data. Create an API token on Kaggle (**Settings → API Tokens**), then:

```bash
pip install -r requirements.txt
kaggle competitions download -c hms-harmful-brain-activity-classification -p data/hms
unzip -q data/hms/hms-harmful-brain-activity-classification.zip -d data/hms

python hms_sklearn_baseline.py --data-dir data/hms --mode quick   # experiments
python hms_sklearn_baseline.py --data-dir data/hms                # full
```

The visible `test.csv` has one example; the real test set is hidden, so scores can only be obtained by
submitting on Kaggle. Local runs are for development and cross-validation.
