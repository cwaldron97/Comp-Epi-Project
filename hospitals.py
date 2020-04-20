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


def write_parsed_data(df, filename):
    with open(filename, "w") as f:
        for row in df.itertuples(index=False, name=None):
            f.write("{}\n".format(row))


# def heatmap(d, bins=(100, 100), smoothing=1.3, cmap="jet"):
#     def getx(pt):
#         return pt.coords[0][0]

#     def gety(pt):
#         return pt.coords[0][1]

#     x = list(d.geometry.apply(getx))
#     y = list(d.geometry.apply(gety))
#     heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
#     extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]

#     logheatmap = np.log(heatmap)
#     logheatmap[np.isneginf(logheatmap)] = 0
#     logheatmap = ndimage.filters.gaussian_filter(logheatmap, smoothing, mode="nearest")

#     plt.imshow(logheatmap, cmap=cmap, extent=extent)
#     plt.colorbar()
#     plt.gca().invert_yaxis()
#     plt.show()


hospitals = read_hospital_data("NYC Hospital Locations Filled.csv")
zipfile = "zip:///home/kevin/code/Comp-Epi-Project/shape/shapefile.zip"
shape = gpd.read_file(zipfile)

boroughs = gpd.read_file(geoplot.datasets.get_path("nyc_boroughs"))


# f, ax = plt.subplots(1)
# shape.plot(ax=ax)
gdf = gpd.GeoDataFrame(
    hospitals,
    geometry=gpd.points_from_xy(hospitals["long"], hospitals["lat"]),
    crs="epsg:4326",
)
# heatmap(gdf, bins=200, smoothing=1.5)

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
