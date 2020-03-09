import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas
from geopy.distance import geodesic


PLOT = False


def process_data(input_file, output_file, headers=["latitude", "longitude"]):
    data = pd.read_csv(input_file, sep="\t", header=None, encoding="ISO-8859-1")
    data.columns = [
        "user_id",
        "venue_id",
        "venue_category_id",
        "venue_category_name",
        "latitude",
        "longitude",
        "tz_offset",
        "utc",
    ]
    data = data.drop(list(set(data.columns) - set(headers)), axis=1)
    data.to_csv(output_file, header=None, index=None, sep=" ")


def read_data(filename):
    df = pd.read_csv(filename, sep=" ", header=None, encoding="ISO-8859-1")
    df.columns = ["latitude", "longitude"]
    return df


def create_gdf(df):
    return geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.longitude, df.latitude)
    )


def plot_data(df, gdf, shape_filename, figure):
    global PLOT
    PLOT = True
    bbox = (
        min(df.longitude) - 0.001,
        min(df.latitude) - 0.001,
        max(df.longitude) + 0.001,
        max(df.latitude) + 0.001,
    )
    shape = geopandas.read_file(shape_filename, bbox=bbox).plot(
        color="white", edgecolor="black"
    )
    gdf.plot(ax=shape, color="red", markersize=0.1)
    plt.figure(figure)


def distance(p1, p2):
    return geodesic(p1, p2).meters


# process_data("data/dataset_TSMC2014_NYC.txt", "data/NYC.txt")
# process_data("data/dataset_TSMC2014_TKY.txt", "data/TKY.txt")

df_NYC = read_data("data/NYC.txt")
df_TKY = read_data("data/TKY.txt")

print(f"NYC Latitude\nMax: {max(df_NYC.latitude)}\tMin:{min(df_NYC.latitude)}\n")
print(f"NYC Longitude\nMax: {max(df_NYC.longitude)}\tMin:{min(df_NYC.longitude)}\n")
print(f"TKY Latitude\nMax: {max(df_TKY.latitude)}\tMin:{min(df_TKY.latitude)}\n")
print(f"TKY Longitude\nMax: {max(df_TKY.longitude)}\tMin:{min(df_TKY.longitude)}\n")

gdf_NYC = create_gdf(df_NYC)
gdf_TKY = create_gdf(df_TKY)

plot_data(df_NYC, gdf_NYC, "data/shapes/gadm36_USA.gpkg", 1)
plot_data(df_TKY, gdf_TKY, "data/shapes/gadm36_JPN.gpkg", 2)

if PLOT:
    plt.show()
