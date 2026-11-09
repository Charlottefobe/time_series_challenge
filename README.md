# HMS – Harmful Brain Activity Classification: scikit-learn baselines

Simple, interpretable baselines for the Kaggle competition
[HMS – Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification).
Each sample is an EEG segment from a critically ill patient; experts voted on six patterns
(seizure, LPD, GPD, LRDA, GRDA, other). Predictions are probability distributions over the six classes and
are scored with the **KL divergence** to the expert-vote distribution (lower is better).

## Files

| File | What it is | Where to run |
|---|---|---|
| `hms_sklearn_baseline.ipynb` | Spectrogram band-power features (88 per EEG), `LogisticRegression` / `HistGradientBoostingClassifier` on soft labels, 5-fold patient-grouped CV | Kaggle notebook |
| `hms_sklearn_baseline.py` | The same method as a command-line script | Your own computer |
| `template.py` | Raw-EEG Welch band-power features, `RandomForestClassifier` on hard labels, one patient-grouped hold-out split | Kaggle or your own computer |
| `requirements.txt` | Python packages for local runs | – |

### Reference scores

| Version | Private KL | Public KL |
|---|---|---|
| `hms_sklearn_baseline.ipynb`, quick mode (3,000 spectrograms, no CV) | 0.9048 | 0.8186 |
| `hms_sklearn_baseline.ipynb`, full mode | to be measured | to be measured |
| 1st place in the competition (deep-learning ensemble) | 0.2723 | – |

## Data

* 26.4 GB, about 28,500 files: `train.csv`, `test.csv`, `train_eegs/`, `train_spectrograms/`,
  `test_eegs/`, `test_spectrograms/`, `sample_submission.csv`.
* Licence: CC BY-NC 4.0. You must join the competition (accept its rules) before you can use or download the data.
* The visible `test.csv` contains **one** example. The real test set is hidden: it is only used when Kaggle
  re-runs a submitted notebook. Scores on the hidden test set can therefore only be obtained on Kaggle;
  local runs are for development and cross-validation.

## Run on Kaggle (recommended)

### `hms_sklearn_baseline.ipynb`

1. Sign in to Kaggle, open the [competition page](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
   and accept the rules (**Join Competition** / **Late Submission**).
2. Go to the competition's **Code** tab and click **New Notebook**. This attaches the competition data.
3. In the editor choose **File → Import Notebook** and upload `hms_sklearn_baseline.ipynb`.
4. In the right-hand panel under **Session options** check that **Accelerator = None** (CPU is enough) and
   **Internet = off** (required for submissions). Wait until the competition data appears under **Input**.
5. Optional: edit the settings in the second cell.
   * `MODE = "full"` uses all training spectrograms (estimated 10 minutes); `MODE = "quick"` uses a random 3,000
     (about 5 minutes).
   * `RUN_CV = False` skips cross-validation (saves about 4 minutes, but no CV score is reported).
6. Click **Save Version → Save & Run All (Commit) → Save**. Follow the progress under **View Active Events**;
   the log shows the CV scores at the end.
7. To submit, open the saved notebook, go to its **Output** tab and click **Submit to Competition**
   (select the version and `submission.csv`). Kaggle re-runs the notebook on the hidden test set; after
   10–20 minutes the private and public scores appear on the competition's **Submissions** page.
   Submissions after the deadline are "late submissions": they are scored but do not appear on the leaderboard.

The notebook finds the data automatically. Depending on the Kaggle environment the competition data is mounted at
`/kaggle/input/competitions/hms-harmful-brain-activity-classification` or
`/kaggle/input/hms-harmful-brain-activity-classification`.

### `template.py`

Create a new notebook from the competition's **Code** tab as above, then use two cells:

```python
%%writefile template.py
# paste the full contents of template.py here
```

```python
!python template.py --data-dir /kaggle/input/competitions/hms-harmful-brain-activity-classification --max-train 2000 --submission
```

Pass `--data-dir` explicitly: the default path in `template.py` is the older mount point, which does not
exist in the current Kaggle environment. `--max-train 0` uses all training EEGs. With `--submission` the script
writes `submission.csv`, which can be submitted as in step 7 above.

## Run on your own computer

Local runs need about 30 GB of free disk space for the data (plus the zip file while unpacking) and a few GB of RAM.

### 1. Install Python packages

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the data

1. Accept the competition rules on Kaggle (see above); downloads are refused otherwise.
2. Create an API token on Kaggle under **Settings → API Tokens** and store it as described on that page
   (for example as `~/.kaggle/kaggle.json` with permissions `600`, or in the `KAGGLE_API_TOKEN` environment variable).
3. Download and unpack:

```bash
mkdir -p data/hms
kaggle competitions download -c hms-harmful-brain-activity-classification -p data/hms
unzip -q data/hms/hms-harmful-brain-activity-classification.zip -d data/hms
rm data/hms/hms-harmful-brain-activity-classification.zip   # optional, frees disk space
```

You can also click **Download All** on the competition's **Data** tab and unzip the file into `data/hms`.
Afterwards `data/hms` should contain `train.csv`, `test.csv`, `train_spectrograms/`, `train_eegs/` and so on.

### 3. Run

```bash
# Spectrogram baseline with 5-fold patient-grouped CV on all training data
python hms_sklearn_baseline.py --data-dir data/hms

# Fast check: random 3,000 spectrograms, no cross-validation
python hms_sklearn_baseline.py --data-dir data/hms --mode quick --no-cv

# Raw-EEG random-forest template
python template.py --data-dir data/hms --max-train 2000 --submission
```

`python hms_sklearn_baseline.py --help` lists all options (`--folds`, `--n-jobs`, `--output`, `--seed`).
Most of the run time is spent reading spectrogram files; on a Kaggle CPU one core reads about 1,000 files per
minute, and the script reads them in parallel on all cores.

## Method of `hms_sklearn_baseline`

1. **Labels**: votes are summed per `eeg_id` and normalised to a probability distribution.
2. **Features**: from the official spectrogram, a 10-minute window around the labelled segment is reduced to
   88 numbers: log band power (delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–20 Hz) for each of the four
   scalp regions (LL, RL, LP, RP) summarised by mean, std, min and max over time; relative band power; and
   left–right asymmetry.
3. **Soft labels**: each EEG is repeated once per class with `sample_weight` equal to the vote fraction, so the
   weighted cross-entropy equals the KL divergence up to a constant.
4. **Validation**: `GroupKFold` by `patient_id`, so the same patient never appears in both the training and the
   validation part. Scores are reported for all EEGs and for EEGs with at least 10 votes, next to a trivial
   model that always predicts the average vote distribution.
5. **Submission**: the model with the better CV score is refitted on all training data and predicts the test set.
