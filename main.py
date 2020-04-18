import pandas as pd
import numpy as np

from models import Simulation, NTAGraphNode, DiseaseModel


def read_data(filename):
    df = pd.read_csv(filename, sep=",", header=None, encoding="ISO-8859-1")
    # Remove extraneous commas/columns after population
    df = df.loc[:, :5]
    # Remove year and county code
    df = df.drop([1, 2], axis=1)
    # Rename columns to use for indexing
    df.columns = ["borough", "nta_code", "nta_name", "population"]
    return df


def read_data2(filename):
    df = pd.read_csv(filename, sep=",", header=None, encoding="ISO-8859-1")
    # Remove extraneous commas/columns after population
    df = df.loc[:, :7]
    # Remove year and county code
    df = df.drop([1, 2], axis=1)
    # Rename columns to use for indexing
    df.columns = ["borough", "nta_code", "nta_name", "population", "lat", "long"]
    return df


def write_parsed_data(df, filename):
    with open(filename, "w") as f:
        for row in df.itertuples(index=False, name=None):
            f.write("{}\n".format(row))


# def create_gdf(df):
#     return geopandas.GeoDataFrame(
#         df, geometry=geopandas.points_from_xy(df.longitude, df.latitude)
#     )


# def plot_data(df, gdf, shape_filename, figure):
#     bbox = (
#         min(df.longitude) - 0.001,
#         min(df.latitude) - 0.001,
#         max(df.longitude) + 0.001,
#         max(df.latitude) + 0.001,
#     )
#     shape = geopandas.read_file(shape_filename, bbox=bbox).plot(
#         color="white", edgecolor="black"
#     )
#     gdf.plot(ax=shape, color="red", markersize=0.1)
#     plt.figure(figure)

# NTAs = read_data("New York Pop NTA updated.csv")
NTAs = read_data2("test-NTAs.csv")
# write_parsed_data(NTAs, "NTAs.txt")

# print(f"NYC: {len(NTAs)}")

# print(f"NYC Latitude\nMax: {max(NTAs.)}\tMin:{min(NTAs.latitude)}\n")
# print(f"NYC Longitude\nMax: {max(NTAs.longitude)}\tMin:{min(NTAs.longitude)}\n")

# gNTAs = create_gdf(NTAs)

# plot_data(NTAs, gNTAs, "data/shapes/gadm36_USA.gpkg", 1)

# plt.show()

s = Simulation(NTAs, None, num_ticks=50)

s.run()
