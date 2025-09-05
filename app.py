import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os



# Import main df
df = pd.read_csv(rf"data/cc_clean.csv")
final_account_labels = pd.read_csv(rf"data/final_acct_table.csv")

# Figure out which columns to bring from final_account_labels
# Always keep acct_num2 (join key)
cols_to_add = [c for c in final_account_labels.columns if c not in df.columns or c == "acct_num2"]

# Merge cleanly
df1 = df.merge(final_account_labels[cols_to_add], on="acct_num2", how="left")


# Convert and add columns

df1['trans_datetime'] = pd.to_datetime(df1['trans_datetime'])
df1['trans_month_year'] = df1['trans_datetime'].dt.to_period('M')
df1['YEAR'] = df1['trans_datetime'].dt.to_period('Y')
df1['category'] = df1['category'].fillna('Unknown')

# Drop duplicate txns potentially from joins
df1 = df1.drop_duplicates(subset=['trans_num'])



st.set_page_config(page_title="My first streamlit project",layout="wide")

# Sidebar
st.sidebar.title("Sprint 1 Project")
menu = st.sidebar.radio("",["Overview","Segment 1","Segment 2","Segment 3"])

data_dir = "data"
image_dir = "images"
plot_dir = "plots"

if menu == "Overview": 
    st.markdown("## Premise")
    st.markdown("## Introduction")
    
    st.markdown("### Original Dataset")
    st.dataframe(df.head(20))


    # Download button for merged dataset
    csv = df1.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv,
        file_name="cleaned_transactions.csv",
        mime="text/csv",
    )

    st.markdown("### Data Preprocessign Pipeline")
    overall_image_path = os.path.join(image_dir, "preprocessing.png")
    st.image(overall_image_path, caption="", use_column_width=True)

    st.markdown("## Customer Segmentation")
    st.markdown("## Proof of Concept")
    overall_image_path = os.path.join(image_dir, "test_screenshot.png")
    st.image(overall_image_path, caption="Mochi", use_column_width=True)
