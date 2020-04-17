from geopy.distance import geodesic
import matplotlib.pyplot as plt


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
        self.history = {
            t: {"S": 0, "E": 0, "I_S": 0, "I_A": 0, "R": 0}
            for t in range(0, num_ticks + 1)
        }
        self.history[0]["S"] = population
        self.history[0]["E"] = 0
        self.history[0]["I_S"] = 0
        self.history[0]["I_A"] = 0
        self.history[0]["R"] = 0

    def update(self, t):
        assert (
            self.history[t]["S"]
            + self.history[t]["E"]
            + self.history[t]["I_S"]
            + self.history[t]["I_A"]
            + self.history[t]["R"]
            == self.population
        )

        b = CONSTANTS["beta"]
        R_b = CONSTANTS["rbeta"]
        e = CONSTANTS["epsilon"]
        pa = CONSTANTS["pa"]
        upt = CONSTANTS["upt"]
        upa = CONSTANTS["upa"]
        u = CONSTANTS["mu"]

        S_t = self.history[t]["S"]
        E_t = self.history[t]["E"]
        I_S_t = self.history[t]["I_S"]
        I_A_t = self.history[t]["I_A"]
        R_t = self.history[t]["R"]

        print(S_t, b, I_A_t, R_b, I_S_t)
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

        self.history[t + 1]["S"] = S_t - new_latent
        self.history[t + 1]["E"] = E_t + new_latent - new_infected
        self.history[t + 1]["I_S"] = (
            I_S_t + new_infected_symptomatic - new_recovered_from_infected_symptomatic
        )
        self.history[t + 1]["I_A"] = (
            I_A_t + new_infected_asymptomatic - new_recovered_from_infected_asymptomatic
        )
        self.history[t + 1]["R"] = R_t + new_recovered

        assert (
            self.history[t + 1]["S"]
            + self.history[t + 1]["E"]
            + self.history[t + 1]["I_S"]
            + self.history[t + 1]["I_A"]
            + self.history[t + 1]["R"]
            == self.population
        )

    def plot_history(self):
        pass


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
        self.model.history[0]["S"] -= num
        self.model.history[0]["I_S"] += num

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
