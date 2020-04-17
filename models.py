from geopy.distance import geodesic
import matplotlib.pyplot as plt
import pandas as pd


CONSTANTS = {
    "beta": 0.8383,
    "epsilon": 1 / 1.1,  # average latency period = 1.1 days
    "mu": 1 / 2.5,  # average infectious period = 2.5 days
    "pa": 0.33,
    "pt": 0.5,
    "rbeta": 0.5,
    "upa": 1 - 0.33,
    "upt": 1 - 0.5,
}


class DiseaseModel:
    def __init__(self, population, num_ticks):
        self.population = population
        self.num_ticks = num_ticks
        self.history = {
            "S": [t for t in range(0, num_ticks + 1)],
            "E": [t for t in range(0, num_ticks + 1)],
            "I_S": [t for t in range(0, num_ticks + 1)],
            "I_A": [t for t in range(0, num_ticks + 1)],
            "R": [t for t in range(0, num_ticks + 1)],
        }
        self.history["S"][0] = population
        self.history["E"][0] = 0
        self.history["I_S"][0] = 0
        self.history["I_A"][0] = 0
        self.history["R"][0] = 0

    def update(self, t):
        assert (
            self.history["S"][t]
            + self.history["E"][t]
            + self.history["I_S"][t]
            + self.history["I_A"][t]
            + self.history["R"][t]
            == self.population
        )

        b = CONSTANTS["beta"]
        R_b = CONSTANTS["rbeta"]
        e = CONSTANTS["epsilon"]
        pa = CONSTANTS["pa"]
        upt = CONSTANTS["upt"]
        upa = CONSTANTS["upa"]
        u = CONSTANTS["mu"]

        S_t = self.history["S"][t]
        E_t = self.history["E"][t]
        I_S_t = self.history["I_S"][t]
        I_A_t = self.history["I_A"][t]
        R_t = self.history["R"][t]

        new_latent = int(S_t / self.population * b * (I_A_t * R_b + I_S_t))
        new_infected_symptomatic = int(E_t * upt * upa)
        new_infected_asymptomatic = int(E_t * e * pa)
        new_infected = new_infected_symptomatic + new_infected_asymptomatic
        new_recovered_from_infected_symptomatic = int(I_S_t * u)
        new_recovered_from_infected_asymptomatic = int(I_A_t * u)
        new_recovered = (
            new_recovered_from_infected_symptomatic
            + new_recovered_from_infected_asymptomatic
        )

        self.history["S"][t + 1] = S_t - new_latent
        self.history["E"][t + 1] = E_t + new_latent - new_infected
        self.history["I_S"][t + 1] = (
            I_S_t + new_infected_symptomatic - new_recovered_from_infected_symptomatic
        )
        self.history["I_A"][t + 1] = (
            I_A_t + new_infected_asymptomatic - new_recovered_from_infected_asymptomatic
        )
        self.history["R"][t + 1] = R_t + new_recovered

        assert (
            self.history["S"][t + 1]
            + self.history["E"][t + 1]
            + self.history["I_S"][t + 1]
            + self.history["I_A"][t + 1]
            + self.history["R"][t + 1]
            == self.population
        )

    def plot_history(self):
        df = pd.DataFrame(self.history)
        t = range(0, self.num_ticks + 1)
        # df.add({"t": range(0, self.num_ticks)})
        plt.plot(
            t,
            df["S"],
            markerfacecolor="blue",
            markersize=6,
            color="skyblue",
            linewidth=2,
            label="Susceptible",
        )
        plt.plot(
            t,
            df["E"],
            markerfacecolor="yellow",
            markersize=6,
            color="yellow",
            linewidth=2,
            label="Latent",
        )
        plt.plot(
            t,
            df["I_S"],
            markerfacecolor="red",
            markersize=6,
            color="red",
            linewidth=2,
            label="Infectious (Symptomatic)",
        )
        plt.plot(
            t,
            df["I_A"],
            markerfacecolor="orange",
            markersize=6,
            color="orange",
            linewidth=2,
            label="Infectious (Asymptomatic)",
        )
        plt.plot(
            t,
            df["R"],
            markerfacecolor="green",
            markersize=6,
            color="green",
            linewidth=2,
            label="Recovered",
        )
        plt.xlabel("Day")
        plt.ylabel("# in Compartment")
        plt.legend()
        plt.show()


class NTAGraphNode:
    def __init__(self, info, num_ticks, seed_num=None):
        # Info looks like ('Borough', 'NTA ID', 'NTA Name', 'Population')
        self.model = DiseaseModel(int(info[3]), num_ticks)
        # Tuple (lat, long)
        # self.centroid = centroid
        self.info = info
        self.neighbors = []

        if seed_num:
            self.seed(seed_num)

    def seed(self, num):
        self.model.history["S"][0] -= num
        self.model.history["I_S"][0] += num

    def distance_in_meters(self, other_point):
        return geodesic(self.center, other_point).meters


class Simulation:
    def __init__(self, nta_data, hospital_data, num_ticks=100):
        self.num_ticks = num_ticks
        self.nodes = self.init_graph_nodes(nta_data)

    def init_graph_nodes(self, df):
        iterator = df.itertuples(index=False, name=None)
        next(iterator)  # skip the first line containing the headers
        nodes = [NTAGraphNode(next(iterator), self.num_ticks, seed_num=100)]
        for row in iterator:
            nodes.append(NTAGraphNode(row, self.num_ticks))
        return nodes

    def run(self):
        # Simulation for n days
        # for i in range(n):
        #     for node in self.nodes:
        #         node.model.update(i)
        for i in range(self.num_ticks):
            self.nodes[0].model.update(i)
        print(self.nodes[0].model.history)
        self.nodes[0].model.plot_history()
