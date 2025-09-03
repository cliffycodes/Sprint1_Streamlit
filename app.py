import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="My first streamlit project",layout="wide")

# Sidebar
st.sidebar.title("Navigation")
menu = st.sidebar.radio("What page should i open",["Test 1","Test 2"])

data_dir = "data"
image_dir = "images"
plot_dir = "plots"

if menu =="Overall" : 
    st.title("Overall Summary")
    st.write("This is an overview of the project")

    overall_image_path = os.path.join(image_dir,"image.png")
    st.image(overall_image_path,caption="Mochi",use_column_width=True)