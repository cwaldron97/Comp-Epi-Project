import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas

df = pd.read_excel(
    "/home/kevin/code/Comp-Epi-Project/data/Census_Demographics_at_the_NYC_City_Council_district__CNCLD__level.xlsx"
)
print(df.head(30))
print(df.tail(30))
