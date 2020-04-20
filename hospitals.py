import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import ndimage
import geoplot

import matplotlib.pylab as pylab

# import geoplot as gp


from models import Simulation, NTAGraphNode, DiseaseModel


def read_hospital_data(filename):
    df = pd.read_csv(filename, sep=",", header=None, encoding="ISO-8859-1")
    # Remove extraneous commas/columns after last column
    df = df.drop([1, 2, 3, 4], axis=1)
    df.columns = [
        "name",
        "lat",
        "long",
        "bed_count",
    ]
    df["lat"] = df.lat.astype(float)
    df["long"] = df.long.astype(float)
    return df


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


def show_kdeplot(shape, gdf):
    ax = geoplot.polyplot(shape, projection=geoplot.crs.AlbersEqualArea(), zorder=1)
    geoplot.kdeplot(
        gdf,
        ax=ax,
        shade=True,
        cmap="Reds",
        n_levels=16,
        shade_lowest=True,
        clip=shape.simplify(0.001, preserve_topology=False),
    )
    geoplot.pointplot(gdf, ax=ax, color="blue")

    plt.show()


NTAs = read_nta_data("New York Pop NTA updated.csv")
hospitals = read_hospital_data("NYC Hospital Locations Filled.csv")
zipfile = "zip:///home/kevin/code/Comp-Epi-Project/shape/shapefile.zip"
shape = gpd.read_file(zipfile)


gdf = gpd.GeoDataFrame(
    hospitals,
    geometry=gpd.points_from_xy(hospitals["long"], hospitals["lat"]),
    crs="epsg:4326",
)


gdf2 = gpd.GeoDataFrame(
    NTAs, geometry=gpd.points_from_xy(NTAs["long"], NTAs["lat"]), crs="epsg:4326",
)
##################
# choropleth WIP #
##################
# NTAs_d = dict(gdf2)
# shape_d = dict(shape)
# for i, ntacode in shape_d["ntacode"].items():
#     indexes = [k for k, v in NTAs_d["nta_code"].items() if ntacode == v]
#     if indexes:
#         index = indexes.pop()
#         NTAs_d["geometry"][index] = shape_d["geometry"][i]
#     else:
#         print(ntacode)

# new_NTAs = pd.DataFrame(NTAs_d)

# gdf3 = gpd.GeoDataFrame(new_NTAs, geometry=new_NTAs.geometry)
# print(gdf3.head(100))

# # show_kdeplot(shape, gdf)
# print(shape.head())
# print(gdf2.head())

# geoplot.polyplot(
#     gdf3,
#     projection=geoplot.crs.AlbersEqualArea(),
#     hue="hospitalization_rate",
#     cmap="Greens",
# )

# plt.show()
