# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# 1. Tytuł
st.title("💡 Fraud Classification App")

# 2. Załaduj model
MODEL_PATH = "model.pkl"
if not os.path.isfile(MODEL_PATH):
    st.error(f"Model not found — umieść '{MODEL_PATH}' w katalogu aplikacji")
    st.stop()

model = joblib.load(MODEL_PATH)

# 3. Wczytywanie danych
st.sidebar.header("Wczytaj dane")
upload_provider = st.sidebar.file_uploader("Plik CSV/Excel z danymi podmiotu ubezpieczającego do predykcji", type=['csv', 'xlsx'])
upload_bene = st.sidebar.file_uploader("Plik CSV/Excel z danymi beneficjenta do predykcji", type=['csv', 'xlsx'])
upload_inpatient = st.sidebar.file_uploader("Plik CSV/Excel z danymi pacjentów przyjętych na oddział do predykcji", type=['csv', 'xlsx'])
upload_outpatient = st.sidebar.file_uploader("Plik CSV/Excel z danymi pacjentów nie przyjetych na oddział do predykcji", type=['csv', 'xlsx'])

if upload_provider and upload_bene and upload_inpatient and upload_outpatient:
    try:
        if upload_provider.name.endswith('.csv'):
            df_provider = pd.read_csv(upload_provider)
        else:
            df_provider = pd.read_excel(upload_provider)
    except Exception as e:
        st.error(f"Błąd odczytu pliku: {e}")
        st.stop()
    try:
        if upload_bene.name.endswith('.csv'):
            df_bene = pd.read_csv(upload_bene)
        else:
            df_bene = pd.read_excel(upload_bene)
    except Exception as e:
        st.error(f"Błąd odczytu pliku: {e}")
        st.stop()
    try:
        if upload_inpatient.name.endswith('.csv'):
            df_inpatient = pd.read_csv(upload_inpatient)
        else:
            df_inptient = pd.read_excel(upload_inpatient)
    except Exception as e:
        st.error(f"Błąd odczytu pliku: {e}")
        st.stop()
    try:
        if upload_outpatient.name.endswith('.csv'):
            df_outpatient = pd.read_csv(upload_outpatient)
        else:
            df_outpatient = pd.read_excel(upload_outpatient)
    except Exception as e:
        st.error(f"Błąd odczytu pliku: {e}")
        st.stop()

    # Łączenie danych w jeden DataFrame
    train_claims = pd.concat([df_inpatient, df_outpatient], axis=0)
    train_claims_with_beneficiary = train_claims.merge(df_bene, on='BeneID', how='left')
    df= train_claims_with_beneficiary.merge(df_provider, on='Provider', how='left')
    
    st.write("🧾 Dane wejściowe:")
    st.dataframe(df.head())

    # Jeśli są kolumny niezgodne z modelem — alert
    expected = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if expected is not None:
        missing = set(expected) - set(df.columns)
        if missing:
            st.warning(f"Brakuje oczekiwanych cech: {missing}")
            st.stop()
        X = df[expected]
    else:
        X = df  # zakładamy poprawny format

    # 4. Predykcja
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    df['predicted_label'] = y_pred
    if y_prob is not None:
        df['probability_not_fraud'] = y_prob[:, 0]
        df['probability_fraud'] = y_prob[:, 1]

    st.write("🎯 Wyniki predykcji:")
    st.dataframe(df)

else:
    st.write("📌 Załaduj dane przez menu po lewej (plik CSV lub Excel).")

# 5. Ręczne wprowadzanie pojedynczego rekordu
st.sidebar.header("Lub wprowadź pojedynczy rekord")
manual = {}
if expected is not None:
    for feat in expected:
        manual[feat] = st.sidebar.text_input(feat, value="")
    if st.sidebar.button("Predict single record"):
        try:
            row = pd.DataFrame([manual], columns=expected).astype(float)
            pred = model.predict(row)[0]
            result = f"⚠️ Fraud" if pred == 1 else "✅ Not Fraud"
            st.sidebar.write(f"Predykcja: **{result}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(row)[0]
                st.sidebar.write(f"Prawdopodobieństwo: not_fraud = {probs[0]:.3f}, fraud = {probs[1]:.3f}")
        except Exception as e:
            st.sidebar.error(f"Błąd: {e}")
