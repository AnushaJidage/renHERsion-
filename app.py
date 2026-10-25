import json, os
from datetime import date, timedelta
import joblib, numpy as np, pandas as pd, streamlit as st
from src.train_model import FEATURES, LABELS, main as train
from src.cycle import analyse, symptom_summary
from src.content import CONDITIONS, SEE_DOCTOR, DISCLAIMER

st.set_page_config(page_title="HerHealth AI", page_icon="💜", layout="wide")

@st.cache_resource
def load():
    if not os.path.exists("models/pcos_rf.joblib"):
        train()
    return joblib.load("models/pcos_rf.joblib"), json.load(open("models/meta.json"))
model, meta = load()

st.title("💜 HerHealth AI")
st.caption("Understand your cycle, spot early warning signs, and walk into the doctor's office prepared.")
st.info(DISCLAIMER + " Your data stays in this session - nothing is stored on a server.")

if "log" not in st.session_state:
    today = date.today()
    st.session_state.log = pd.DataFrame({
        "date": [today - timedelta(days=d) for d in (90, 61, 32, 2)],
        "cramps": [1, 0, 1, 1], "acne": [0, 1, 0, 1], "fatigue": [0, 0, 1, 1],
        "heavy_flow": [0, 0, 0, 1], "mood": [3, 4, 3, 2], "pain": [4, 2, 5, 6]})

t1, t2, t3, t4, t5 = st.tabs(["🩺 Risk Screening", "📅 Cycle Tracker", "📊 Insights", "📄 Doctor Report", "📚 Learn"])

with t1:
    st.subheader("PCOS risk screening")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 12, 60, 24)
    bmi = c1.number_input("BMI", 14.0, 50.0, 23.0, 0.1)
    cl = c2.number_input("Average cycle length (days)", 15, 120, 30)
    cpy = c2.number_input("Periods in the last 12 months", 0, 15, 11)
    ff = c3.slider("Fast-food days / week", 0, 7, 2)
    ex = c3.slider("Exercise days / week", 0, 7, 3)
    s1, s2, s3, s4, s5 = st.columns(5)
    wg = s1.checkbox("Weight gain"); hg = s2.checkbox("Excess hair")
    ac = s3.checkbox("Persistent acne"); sk = s4.checkbox("Skin darkening"); hl = s5.checkbox("Hair thinning")
    if st.button("Assess risk", type="primary"):
        x = pd.DataFrame([[age, bmi, cl, cpy, int(wg), int(hg), int(ac), int(sk), int(hl), ff, ex]], columns=FEATURES)
        p = float(model.predict_proba(x)[0, 1])
        band = "Low" if p < .33 else "Moderate" if p < .66 else "High"
        st.session_state.risk = (p, band, x.iloc[0].to_dict())
    if "risk" in st.session_state:
        p, band, vals = st.session_state.risk
        st.metric("Screening risk score", f"{p*100:.0f}%", band)
        st.progress(min(p, 1.0))
        # simple explanation: importance x whether the value looks atypical
        flags = []
        if vals["cycle_length_days"] > 35 or vals["cycle_length_days"] < 21: flags.append("Cycle length outside 21-35 days")
        if vals["cycles_per_year"] < 9: flags.append("Fewer than 9 periods per year")
        if vals["bmi"] >= 25: flags.append("BMI in overweight range")
        for k in ("weight_gain", "hair_growth", "acne", "skin_darkening", "hair_loss"):
            if vals[k]: flags.append(LABELS[k])
        if flags:
            st.write("**Factors contributing:**"); [st.write(f"- {f}") for f in flags]
        if band != "Low":
            st.warning("Consider discussing these results with a gynaecologist. PCOS is diagnosed with clinical evaluation and tests, not by this tool.")
        else:
            st.success("Low screening risk. Keep tracking and see a doctor if symptoms change.")
        with st.expander("About the model"):
            st.write(f"Random Forest, trained on **{meta['data']}** data, ROC-AUC {meta['roc_auc']}.")
            imp = pd.Series(meta["importances"]).rename(index=LABELS).sort_values()
            st.bar_chart(imp)

with t2:
    st.subheader("Log your periods & symptoms")
    st.caption("Add a row for each period start. 1 = yes, 0 = no. Mood 1-5, pain 0-10.")
    st.session_state.log = st.data_editor(st.session_state.log, num_rows="dynamic", use_container_width=True,
        column_config={"date": st.column_config.DateColumn("Period start")})
    st.download_button("⬇️ Export my log (CSV)", st.session_state.log.to_csv(index=False), "my_cycle_log.csv")

log = st.session_state.log.dropna(subset=["date"])
res = analyse(log["date"].tolist())

with t3:
    st.subheader("Your cycle insights")
    if not res["ok"]:
        st.info(res["msg"])
    else:
        a, b, c = st.columns(3)
        a.metric("Average cycle", f"{res['avg']} days"); b.metric("Variability (±)", f"{res['std']} days")
        c.metric("Next period (est.)", str(res["next"]))
        st.write(f"Estimated ovulation window around **{res['ovulation']}** (estimate only; not for contraception).")
        st.line_chart(pd.DataFrame({"Cycle length (days)": res["gaps"]}))
        if res["flags"]:
            st.warning("**Patterns worth discussing with a doctor:**\n" + "\n".join(f"- {f}" for f in res["flags"]))
        else:
            st.success("Your logged cycles look within typical ranges.")
    summ = symptom_summary(log)
    if len(summ):
        st.write("**Most frequent symptoms**"); st.bar_chart(summ)
    if len(log) > 1:
        st.write("**Pain & mood over time**")
        st.line_chart(log.set_index("date")[["pain", "mood"]])

with t4:
    st.subheader("Clinic-ready summary")
    lines = [f"HerHealth AI summary - {date.today()}", ""]
    if res["ok"]:
        lines += [f"Average cycle: {res['avg']} days (variability ±{res['std']})",
                  f"Cycle gaps (days): {res['gaps']}",
                  "Flags: " + ("; ".join(res["flags"]) or "none")]
    else:
        lines.append("Not enough cycle data logged yet.")
    summ = symptom_summary(log)
    lines.append("Symptoms logged: " + (", ".join(f"{k} x{int(v)}" for k, v in summ.items()) or "none"))
    if "risk" in st.session_state:
        p, band, _ = st.session_state.risk
        lines.append(f"PCOS screening score: {p*100:.0f}% ({band}) - screening only, not a diagnosis")
    lines += ["", "Questions to ask my doctor:", "- Could my cycle pattern indicate PCOS/thyroid issues?",
              "- Which tests (hormone panel, ultrasound, TSH, haemoglobin) do you recommend?", "", DISCLAIMER]
    report = "\n".join(lines)
    st.text_area("Preview", report, height=320)
    st.download_button("⬇️ Download report", report, "herhealth_report.txt")

with t5:
    st.subheader("Know your body")
    for k, v in CONDITIONS.items():
        with st.expander(k): st.write(v)
    st.write("**See a doctor if:**"); [st.write(f"- {s}") for s in SEE_DOCTOR]
