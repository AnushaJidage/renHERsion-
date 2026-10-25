"""Cycle-tracking analytics."""
import pandas as pd

def analyse(starts):
    """starts: list of period start dates. Returns dict of insights."""
    s = pd.to_datetime(pd.Series(starts)).dropna().sort_values().reset_index(drop=True)
    if len(s) < 3:
        return {"ok": False, "msg": "Log at least 3 period start dates to see insights."}
    gaps = s.diff().dt.days.dropna()
    avg, sd = gaps.mean(), gaps.std()
    flags = []
    if avg < 21: flags.append("Cycles shorter than 21 days on average")
    if avg > 35: flags.append("Cycles longer than 35 days on average")
    if sd > 7: flags.append("Cycle length varies by more than a week")
    if gaps.max() > 60: flags.append("A gap of 60+ days between periods")
    return {"ok": True, "avg": round(avg, 1), "std": round(sd, 1), "gaps": gaps.astype(int).tolist(),
            "next": (s.iloc[-1] + pd.Timedelta(days=round(avg))).date(),
            "ovulation": (s.iloc[-1] + pd.Timedelta(days=round(avg) - 14)).date(),
            "flags": flags}

def symptom_summary(df):
    cols = [c for c in df.columns if c not in ("date", "mood", "pain")]
    out = df[cols].sum().sort_values(ascending=False)
    return out[out > 0]
