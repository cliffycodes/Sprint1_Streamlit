import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os


# Import main df
df = pd.read_csv(rf"C:\Users\221947\OneDrive - Security Bank Corporation\Desktop\Cliffy_Environment\Practiz\cc_clean.csv")
final_account_labels = pd.read_csv(rf"final_acct_table.csv")

# Join tables w/ account labels
df1 = pd.merge(df,final_account_labels,on='acct_num2',how='left')
# Convert and add columns

df1['trans_datetime'] = pd.to_datetime(df1['trans_datetime'])
df1['trans_month_year'] = df1['trans_datetime'].dt.to_period('M')
df1['YEAR'] = df1['trans_datetime'].dt.to_period('Y')
df1['category'] = df1['category'].fillna('Unknown')

# Drop duplicate txns potentially from joins
df1 = df1.drop_duplicates(subset=[''])



st.set_page_config(page_title="My first streamlit project",layout="wide")

# Sidebar
st.sidebar.title("Sprint 1 Project")
menu = st.sidebar.radio("",["Overview","Segment 1","Segment 2","Segment 3"])

data_dir = "data"
image_dir = "images"
plot_dir = "plots"

if menu =="Overview" : 
    st.markdown("# Premise")
    st.markdown("## Introduction")
    
    # Original table
    st.markdown("### Here's the original table")
    st.table(df)

    st.markdown("## Customer Segmentation")
    st.markdown("## Proof of Concept")
    overall_image_path = os.path.join(image_dir,"test_screenshot.png")
    st.image(overall_image_path,caption="Mochi",use_column_width=True)