# 💜 HerHealth AI - recHERsion'26 (PS1: Women's Healthcare)

A privacy-first web app that helps women **understand and manage their health**:

| Feature | Addresses PS1 challenge |
|---|---|
| PCOS risk screening (Random Forest + explanation of factors) | Gaps in diagnosis |
| Cycle & symptom tracker with irregularity flags, next-period & ovulation estimates | Symptom tracking |
| Insights dashboard (cycle variability, symptom frequency, pain/mood trends) | Meaningful clinical insights |
| One-click clinic-ready report + questions for the doctor | Access to clinical care |
| Learn hub (PCOS, endometriosis, anaemia, thyroid, breast & cervical health) | Health awareness |

**Privacy by design:** no accounts, no server-side storage - data lives in the browser session and can be exported as CSV.

## Run locally
```bash
git clone https://github.com/<your-username>/herhealth-ai.git
cd herhealth-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train_model        # trains model (auto-runs on first launch too)
streamlit run app.py
```

## Use a real dataset
Download the Kaggle *PCOS* dataset, map columns to `FEATURES` in `src/train_model.py`, save as `data/pcos.csv` (with binary `label`), then `python -m src.train_model --real`.

## Project structure
```
app.py              Streamlit UI (5 tabs)
src/train_model.py  data + Random Forest training
src/cycle.py        cycle analytics
src/content.py      awareness content
models/             saved model + metrics
```

## Tech stack
Python, Streamlit, scikit-learn, pandas, joblib.

## Limitations & ethics
Educational screening aid - **not a diagnosis**. The bundled model is trained on synthetic data for demo; validate on real clinical data before any real-world use. Cycle/ovulation estimates are not contraception.

## Future work
Multilingual UI (Kannada/Hindi), PDF reports, wearable data, anonymised clinician dashboard, more conditions (thyroid, anaemia screening).

Built by Anusha Jidage for recHERsion'26.
