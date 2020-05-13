import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import ndimage
import geoplot

import matplotlib.pylab as pylab

from pprint import pprint

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

shape_ids = list(zip(shape["ntacode"], shape["ntaname"]))
nta_ids = set(NTAs["nta_code"])
missing = {i[0]: i[1] for i in shape_ids if i[0] not in nta_ids}

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

results = dict(
    pd.read_csv("simulation-results_COVID_100_10_25%.csv", sep="|", header=None)
)

num_ticks = len(eval(results[1][0])["S"])

new_NTAs["results"] = np.nan
for i in range(num_ticks):
    new_NTAs[f"day_{i}"] = np.nan
new_NTAs = dict(new_NTAs)

for index, ntacode in results[0].items():
    found = [i for i, v in new_NTAs["nta_code"].items() if ntacode == v]
    if found:
        assert len(found) == 1
        loc = found.pop()
        new_NTAs["results"][loc] = results[1][index]
        for i in range(num_ticks):
            new_NTAs[f"day_{i}"][loc] = eval(results[1][index])["I_S"][i]
    else:
        continue
        # print(ntacode, results[0][index])

NTAs_with_results = pd.DataFrame(new_NTAs)

gdf4 = gpd.GeoDataFrame(NTAs_with_results, geometry=NTAs_with_results.geometry)

import mapclassify as mc

m = 0
max_ind = 0
for i in range(num_ticks):
    # new_m = max(gdf4[f"day_{i}"])
    new_m = sum(gdf4[f"day_{i}"])
    if new_m > m:
        m = new_m
        max_ind = i

print(f"Peak Infected (Symptomatic): {m}\t Day: {max_ind}")

proj = geoplot.crs.AlbersEqualArea()

# bins = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# scheme_bracket = f"day_{int(num_ticks / 2)}"
# scheme = mc.UserDefined(gdf4[f"day_30"], bins)
scheme = mc.EqualInterval(gdf4[f"day_{max_ind}"], k=16)

total_resilient = 0
total_susceptible = 0
total_population = 0
for nta_results in gdf4["results"]:
    total_resilient += eval(nta_results)["R"][num_ticks - 1]
    total_susceptible += eval(nta_results)["S"][num_ticks - 1]
    total_population += (
        eval(nta_results)["S"][0]
        + eval(nta_results)["I_S"][0]
        + eval(nta_results)["I_A"][0]
        + eval(nta_results)["E"][0]
        + eval(nta_results)["R"][0]
    )
# print(f"Total Suscepible: {total_susceptible}")
# print(f"Total Resilient: {total_resilient}")
print(
    f"Population of NYC Infected by Day {num_ticks}: {total_resilient / total_population}"
)

for i in range(num_ticks):
    geoplot.choropleth(
        gdf4,
        hue=f"day_{i}",
        linewidth=0.5,
        scheme=scheme,
        legend=True,
        legend_kwargs={"loc": "upper left", "fontsize": "xx-large"},
        projection=proj,
        figsize=(16, 16),
        edgecolor="black",
    )
    plt.title(f"Day {i+1}", fontsize=36)

    # plt.subplots_adjust(top=0.75)
    # plt.subplots_adjust(bottom=0.03)
    # plt.subplots_adjust(left=0.03)
    # plt.subplots_adjust(right=0.75)
    # plt.subplots_adjust(hspace=0.03)
    # plt.subplots_adjust(wspace=0.03)
    # plt.suptitle("Size of 'Infectious (Symptomatic)' Compartment Over Time", fontsize=14)
    fig = plt.gcf()
    if i < 9:
        plt.savefig(f"choropleth/day_0{i+1}.png", bbox_inches="tight", pad_inches=0.1)
    else:
        plt.savefig(f"choropleth/day_{i+1}.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
