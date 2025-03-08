import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from copy import copy
from PIL import Image
import requests
from io import BytesIO

col = st.columns((4,4), gap='medium')

with col[0]:
    player0 = st.selectbox(
        "Select file:",
        ['test'],
        index=None, #default value for user not come to an empty page, required
        key='p0'
    )