from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from app.models import Category, Transaction

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _extract_month(question: str) -> Tuple[int, int]:
    lowered = question.lower()
    for name, month_num in MONTHS.items():
        if name in lowered:
            year = datetime.utcnow().year
            match = re.search(r"(20\d{2})", lowered)
            if match:
                year = int(match.group(1))
            return year, month_num
    return 0, 0


def _intent(question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ["spike", "increase", "higher", "jump", "rose"]):
        return "spending_spike"
    if "how much" in lowered and "spend" in lowered:
        return "spend_on"
    if any(word in lowered for word in ["largest", "biggest"]) and any(
        word in lowered for word in ["purchase", "transaction"]
    ):
        return "largest_purchases"
    return "unknown"


def _df_from_transactions(transactions: List[Transaction]) -> pd.DataFrame:
    data = [
        {
            "date": t.date,
            "description": t.description,
            "amount": t.amount,
            "category": t.category or "Misc",
            "merchant": t.merchant or "",
        }
        for t in transactions
    ]
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def _spike_insight(df: pd.DataFrame, question: str) -> Tuple[Dict[str, object], str]:
    year, month = _extract_month(question)
    if df.empty:
        return {"message": "No transactions available"}, "No transactions available."

    if year == 0:
        target_month = df["month"].max()
    else:
        target_month = pd.Timestamp(year=year, month=month, day=1)

    prev_month = (target_month - pd.DateOffset(months=1)).normalize()
    current = df[df["month"] == target_month]
    previous = df[df["month"] == prev_month]

    current_expenses = current[current["amount"] < 0]
    prev_expenses = previous[previous["amount"] < 0]

    current_cat = current_expenses.groupby("category")["amount"].sum().abs()
    prev_cat = prev_expenses.groupby("category")["amount"].sum().abs()
    delta_cat = (current_cat - prev_cat).fillna(current_cat).sort_values(ascending=False).head(5)

    current_merch = current_expenses.groupby("merchant")["amount"].sum().abs()
    prev_merch = prev_expenses.groupby("merchant")["amount"].sum().abs()
    delta_merch = (current_merch - prev_merch).fillna(current_merch).sort_values(ascending=False).head(5)

    data = {
        "month": target_month.strftime("%Y-%m"),
        "top_categories": delta_cat.dropna().to_dict(),
        "top_merchants": delta_merch.dropna().to_dict(),
    }
    summary = (
        f"Top categories driving the spike in {data['month']}: "
        f"{', '.join(list(data['top_categories'].keys())[:3]) or 'none'}."
    )
    return data, summary


def _spend_on_insight(df: pd.DataFrame, question: str, categories: List[str]) -> Tuple[Dict[str, object], str]:
    lowered = question.lower()
    match = re.search(r"spend on (.+)", lowered)
    if not match:
        match = re.search(r"spent on (.+)", lowered)
    target = match.group(1).strip() if match else ""
    target = re.sub(r"[\?\.\!]", "", target).strip()

    if df.empty:
        return {"message": "No transactions available"}, "No transactions available."

    category = None
    for cat in categories:
        if cat.lower() in target:
            category = cat
            break

    if category:
        subset = df[(df["category"].str.lower() == category.lower()) & (df["amount"] < 0)]
        total = subset["amount"].sum() * -1
        data = {"category": category, "total_spend": round(float(total), 2)}
        summary = f"You spent ${data['total_spend']:.2f} on {category}."
        return data, summary

    merchant_subset = df[df["merchant"].str.lower().str.contains(target)]
    total = merchant_subset[merchant_subset["amount"] < 0]["amount"].sum() * -1
    data = {"merchant": target, "total_spend": round(float(total), 2)}
    summary = f"You spent ${data['total_spend']:.2f} on {target}."
    return data, summary


def _largest_purchases(df: pd.DataFrame, question: str) -> Tuple[Dict[str, object], str]:
    if df.empty:
        return {"message": "No transactions available"}, "No transactions available."

    match = re.search(r"(\d+)", question)
    limit = int(match.group(1)) if match else 5
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()
    top = expenses.sort_values("abs_amount", ascending=False).head(limit)
    records = top[["date", "description", "amount", "category", "merchant"]].to_dict(
        orient="records"
    )
    data = {"count": len(records), "transactions": records}
    summary = f"Top {data['count']} largest purchases returned."
    return data, summary


def _refine_summary(question: str, data: Dict[str, object], summary: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return summary
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = (
            "You are a finance assistant. Provide a concise, plain-English summary "
            "based on the data. Avoid new facts.\n\n"
            f"Question: {question}\n"
            f"Data: {data}\n"
            f"Draft summary: {summary}\n"
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return summary


def run_insights(db, question: str) -> Dict[str, object]:
    transactions = db.query(Transaction).all()
    df = _df_from_transactions(transactions)
    categories = [c.name for c in db.query(Category).all()]
    intent = _intent(question)

    if intent == "spending_spike":
        data, summary = _spike_insight(df, question)
    elif intent == "spend_on":
        data, summary = _spend_on_insight(df, question, categories)
    elif intent == "largest_purchases":
        data, summary = _largest_purchases(df, question)
    else:
        data = {"message": "Could not parse intent"}
        summary = "Try asking about spending spikes, category totals, or largest purchases."

    refined = _refine_summary(question, data, summary)
    return {"intent": intent, "data": data, "summary": refined}
