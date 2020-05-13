import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


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


def read_hospital_data(filename):
    df = pd.read_csv(filename, sep=",", header=None, encoding="ISO-8859-1")
    # Remove extraneous commas/columns after last column
    df = df.drop([1, 2, 3, 4], axis=1)
    df.columns = [
        "name",
        "long",
        "lat",
        "bed_count",
    ]
    df["lat"] = df.lat.astype(float)
    df["long"] = df.long.astype(float)
    return df


# def write_parsed_data(df, filename):
#     with open(filename, "w") as f:
#         for row in df.itertuples(index=False, name=None):
#             f.write("{}\n".format(row))


NTAs = read_nta_data("New York Pop NTA updated.csv")
hospitals = read_hospital_data("NYC Hospital Locations Filled.csv")

# for restriction in [1.0, 0.75, 0.5, 0.25]:
# for restriction in [1.0, 0.25]:
for restriction in [1.0, 0.75, 0.5, 0.25]:
    s = Simulation(NTAs, hospitals)
    s.run(restriction)

# for n_id, n in s.nodes.items():
#     pop = n.model.population
#     day20 = n.model.history["R"][20] / pop
#     day40 = n.model.history["R"][40] / pop
#     day60 = n.model.history["R"][60] / pop
#     day80 = n.model.history["R"][80] / pop
#     print(
#         "{}:\n\t20: {}\n\t40: {}\n\t60: {}\n\t80: {}\n".format(
#             n_id, day20, day40, day60, day80
#         )
#     )
