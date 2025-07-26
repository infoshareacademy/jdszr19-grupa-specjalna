import streamlit as st
import pandas as pd

st.write("Greetings, earthings, we have now taken over your radio!")

df = pd.DataFrame({
  'height': [211, 203, 206],
  'name': ['Kovalov', 'Vucic', 'Karnik']
})

st.dataframe(df.style.highlight_max(axis=0))

df