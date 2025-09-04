import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="My first streamlit project",layout="wide")

# Sidebar
st.sidebar.title("Sprint 1 Project")
menu = st.sidebar.radio("",["Overview","Segment 1","Segment 2","Segment 3"])

data_dir = "data"
image_dir = "images"
plot_dir = "plots"

if menu =="Overview" : 
    st.markdown("# Title")
    st.markdown("## Introduction")
    st.markdown("## Customer Segmentation")
    st.markdown("## Proof of Concept")
    overall_image_path = os.path.join(image_dir,"test_screenshot.png")
    st.image(overall_image_path,caption="Mochi",use_column_width=True)