import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import ndimage
import geoplot

import matplotlib.pylab as pylab

# import geoplot as gp


from models import Simulation, NTAGraphNode, DiseaseModel


def read_nta_data(filename):
    df = pd.read_csv(filename, sep=",", header=None, encoding="ISO-8859-1")
    # Remove extraneous commas/columns after last column
    df = df.loc[:, :6]
    # Rename columns to use for indexing
    df.columns = [
        "borough",
        "nta_code",
        "nta_name",
        "population",
        "lat",
        "long",
        "hospitalization_rate",
    ]
    df["lat"] = df.lat.astype(float)
    df["long"] = df.long.astype(float)
    df["hospitalization_rate"] = df.hospitalization_rate.astype(float)
    return df


def write_parsed_data(df, filename):
    with open(filename, "w") as f:
        for row in df.itertuples(index=False, name=None):
            f.write("{}\n".format(row))


NTAs = read_nta_data("New York Pop NTA updated.csv")
zipfile = "zip:///home/kevin/code/Comp-Epi-Project/shape/shapefile.zip"
shape = gpd.read_file(zipfile)

gdf2 = gpd.GeoDataFrame(
    NTAs, geometry=gpd.points_from_xy(NTAs["long"], NTAs["lat"]), crs="epsg:4326",
)

NTAs_d = dict(gdf2)
shape_d = dict(shape)
for i, ntacode in NTAs_d["nta_code"].items():
    indexes = [k for k, v in shape_d["ntacode"].items() if ntacode == v]
    if indexes:
        assert len(indexes) == 1
        index = indexes.pop()
        NTAs_d["geometry"][i] = shape_d["geometry"][index]
    else:
        print(ntacode, shape_d["ntaname"][i])

new_NTAs = pd.DataFrame(NTAs_d)

gdf3 = gpd.GeoDataFrame(new_NTAs, geometry=new_NTAs.geometry)

results = dict(pd.read_csv("simulation-results.csv", sep="|", header=None))

BRACKET1 = "day_30"
BRACKET2 = "day_60"
BRACKET3 = "day_90"
BRACKET4 = "day_120"
NUM1 = int(BRACKET1.split("_")[1])
NUM2 = int(BRACKET2.split("_")[1])
NUM3 = int(BRACKET3.split("_")[1])
NUM4 = int(BRACKET4.split("_")[1])

new_NTAs["results"] = np.nan
new_NTAs[BRACKET1] = np.nan
new_NTAs[BRACKET2] = np.nan
new_NTAs[BRACKET3] = np.nan
new_NTAs[BRACKET4] = np.nan
new_NTAs = dict(new_NTAs)


for index, ntacode in results[0].items():
    found = [i for i, v in new_NTAs["nta_code"].items() if ntacode == v]
    if found:
        assert len(found) == 1
        loc = found.pop()
        new_NTAs["results"][loc] = results[1][index]
        new_NTAs[BRACKET1][loc] = eval(results[1][index])["R"][NUM1]
        new_NTAs[BRACKET2][loc] = eval(results[1][index])["R"][NUM2]
        new_NTAs[BRACKET3][loc] = eval(results[1][index])["R"][NUM3]
        new_NTAs[BRACKET4][loc] = eval(results[1][index])["R"][NUM4]
    else:
        print(ntacode, results[0][index])

NTAs_with_results = pd.DataFrame(new_NTAs)

gdf4 = gpd.GeoDataFrame(NTAs_with_results, geometry=NTAs_with_results.geometry)

import mapclassify as mc

scheme = mc.EqualInterval(gdf4[BRACKET4], k=8)
proj = geoplot.crs.AlbersEqualArea()
fig, axarr = plt.subplots(2, 2, figsize=(16, 16), subplot_kw={"projection": proj})

geoplot.choropleth(
    gdf4, hue=BRACKET1, linewidth=1, scheme=scheme, ax=axarr[0][0], legend=True,
)
axarr[0][0].set_title("Day {}".format(NUM1), fontsize=12)

geoplot.choropleth(
    gdf4, hue=BRACKET2, linewidth=1, scheme=scheme, ax=axarr[0][1], legend=True,
)
axarr[0][1].set_title("Day {}".format(NUM2), fontsize=12)

geoplot.choropleth(
    gdf4, hue=BRACKET3, linewidth=1, scheme=scheme, ax=axarr[1][0], legend=True,
)
axarr[1][0].set_title("Day {}".format(NUM3), fontsize=12)

geoplot.choropleth(
    gdf4, hue=BRACKET4, linewidth=1, scheme=scheme, ax=axarr[1][1], legend=True,
)
axarr[1][1].set_title("Day {}".format(NUM4), fontsize=12)

plt.subplots_adjust(top=0.85)
plt.subplots_adjust(bottom=0.01)
plt.subplots_adjust(left=0.01)
plt.subplots_adjust(right=0.85)
plt.subplots_adjust(hspace=0.01)
plt.subplots_adjust(wspace=0.01)
plt.suptitle("Size of 'Recovered' Compartment Over Time", fontsize=16)

fig = plt.gcf()
plt.savefig("quad-chloropleth.png", bbox_inches="tight", pad_inches=0.1)

# geoplot.choropleth(
#     gdf3,
#     projection=geoplot.crs.AlbersEqualArea(),
#     hue="hospitalization_rate",
#     cmap="Greens",
#     legend=True,
#     edgecolor="black",
#     linewidth=1,
# )

plt.show()
