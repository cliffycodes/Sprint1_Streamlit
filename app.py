import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="My first streamlit project",layout="wide")

# Sidebar
st.sidebar.title("Navigation")
menu = st.sidebar.radio("What page should i open",["Test 1","Test 2"])