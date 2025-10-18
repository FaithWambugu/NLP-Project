import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import wandb
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

wandb.init(
    project="sentiment_analysis_distilbert",
    name="streamlit_inference",
    config={
        "binary_model_path": "saved_models/binary_distilbert",
        "multiclass_model_path": "saved_models/multiclass_distilbert",
        "batch_size": 16,
        "max_len": 160
    }
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "saved_models/binary_distilbert"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

MODEL_PATH_MULTI = "saved_models/multiclass_distilbert"
tokenizer_multi = DistilBertTokenizer.from_pretrained(MODEL_PATH_MULTI)
model_multi = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH_MULTI)
device_multi = device  
model_multi.to(device_multi)
model_multi.eval()

st.sidebar.header("Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV file with tweets", type=["csv"])
selected_brand = st.sidebar.selectbox("Select Brand", ["All", "Apple", "Google"])

def batch_predict_binary(texts, batch_size=16):
    preds_all, probs_neg_all = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Binary Predictions"):
        batch = texts[i:i+batch_size]
        encoding = tokenizer(
            batch, truncation=True, padding=True, max_length=160, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            probs_neg = probs[:,0]

        preds_all.extend(preds)
        probs_neg_all.extend(probs_neg)

        # log batch to WandB
        for t, p, pn in zip(batch, preds, probs_neg):
            wandb.log({"binary_prediction": p, "negative_prob": pn, "text": t})

    return np.array(preds_all), np.array(probs_neg_all)

def batch_predict_multiclass(texts, batch_size=16):
    preds_all = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Multiclass Predictions"):
        batch = texts[i:i+batch_size]
        encoding = tokenizer_multi(batch, truncation=True, padding=True,
                                   max_length=160, return_tensors="pt").to(device_multi)
        with torch.no_grad():
            outputs = model_multi(**encoding)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds_all.extend(preds)

        # log batch to WandB
        for t, p in zip(batch, preds):
            wandb.log({"multiclass_prediction": p, "text": t})

    return np.array(preds_all)

st.title("Sentiment Analysis App with WandB Logging")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ["brand", "text"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
    else:
        apple_products = ["iPad", "Apple", "iPad or iPhone App", "iPhone", "Other Apple product or service"]
        google_products = ["Google", "Other Google product or service", "Android", "Android App"]
        df["parent_brand"] = df["brand"].apply(
            lambda x: "Apple" if x in apple_products else ("Google" if x in google_products else "Other")
        )

        if selected_brand != "All":
            df = df[df["parent_brand"] == selected_brand]

        st.success(f"Loaded {len(df)} tweets after filtering.")

        st.subheader("🚀 Running Predictions")
        preds_bin, probs_neg = batch_predict_binary(df["text"].tolist(), batch_size=16)
        df["binary_label"] = preds_bin
        df["negative_prob"] = probs_neg
        df["binary_sentiment"] = df["binary_label"].apply(lambda x: "Negative" if x==0 else "Positive")

        preds_multi = batch_predict_multiclass(df["text"].tolist(), batch_size=16)
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        df["multi_label"] = preds_multi
        df["multi_sentiment"] = df["multi_label"].map(label_map)

        st.subheader("📊 Multiclass Sentiment Trends")
        trend_df = df.groupby(["parent_brand", "multi_sentiment"]).size().unstack(fill_value=0)
        st.bar_chart(trend_df)
        wandb.log({"multiclass_trends": wandb.Table(dataframe=trend_df)})
        st.caption("Sentiment distribution by brand (Positive / Neutral / Negative)")   
        
        st.subheader("🚨 Negative Tweets")
        negative_alerts = df[df["binary_sentiment"]=="Negative"].sort_values(by="negative_prob", ascending=False)
        st.dataframe(
            negative_alerts[["brand","text","negative_prob"]].style.applymap(
                lambda x: "background-color: #FFCCCC" if isinstance(x,(float,int)) and x>0.7 else ""
            )
        )
