import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


from models import Simulation, NTAGraphNode, DiseaseModel


def read_data(filename):
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
    return df


def write_parsed_data(df, filename):
    with open(filename, "w") as f:
        for row in df.itertuples(index=False, name=None):
            f.write("{}\n".format(row))


NTAs = read_data("New York Pop NTA updated.csv")

s = Simulation(NTAs, None, num_ticks=81)

s.run()

for n_id, n in s.nodes.items():
    pop = n.model.population
    day20 = n.model.history["R"][20] / pop
    day40 = n.model.history["R"][40] / pop
    day60 = n.model.history["R"][60] / pop
    day80 = n.model.history["R"][80] / pop
    print(
        "{}:\n\t20: {}\n\t40: {}\n\t60: {}\n\t80: {}\n".format(
            n_id, day20, day40, day60, day80
        )
    )
