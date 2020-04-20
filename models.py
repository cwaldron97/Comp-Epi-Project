from geopy.distance import geodesic
import matplotlib.pyplot as plt
import pandas as pd

from pprint import pprint, pformat
from decimal import *
from numpy.random import rand


CONSTANTS = {
    "beta": Decimal(0.8383),
    "epsilon": Decimal(1) / Decimal(1.1),  # average latency period = 1.1 days
    "mu": Decimal(1) / Decimal(2.5),  # average infectious period = 2.5 days
    "pa": Decimal(0.33),
    "pt": Decimal(0.5),
    "rbeta": Decimal(0.5),
    "upa": Decimal(1 - 0.33),
    "upt": Decimal(1 - 0.5),
}


def make_compartment_charts(df, num_ticks, ax, title):
    t = range(0, num_ticks + 1)
    # df.add({"t": range(0, self.num_ticks)})
    ax.plot(
        t,
        df["S"],
        markerfacecolor="blue",
        markersize=6,
        color="skyblue",
        linewidth=2,
        label="Susceptible",
    )
    ax.plot(
        t,
        df["E"],
        markerfacecolor="yellow",
        markersize=6,
        color="yellow",
        linewidth=2,
        label="Latent",
    )
    ax.plot(
        t,
        df["I_S"],
        markerfacecolor="red",
        markersize=6,
        color="red",
        linewidth=2,
        label="Infectious (Symptomatic)",
    )
    ax.plot(
        t,
        df["I_A"],
        markerfacecolor="orange",
        markersize=6,
        color="orange",
        linewidth=2,
        label="Infectious (Asymptomatic)",
    )
    ax.plot(
        t,
        df["R"],
        markerfacecolor="green",
        markersize=6,
        color="green",
        linewidth=2,
        label="Recovered",
    )
    # plt.xlabel("Day")
    # plt.ylabel("# in Compartment")
    ax.legend()
    ax.set_title(title)


class DiseaseModel:
    def __init__(self, population, num_ticks):
        self.population = Decimal(population)
        self.num_ticks = num_ticks
        self.history = {
            "S": [Decimal(0) for t in range(0, num_ticks + 1)],
            "E": [Decimal(0) for t in range(0, num_ticks + 1)],
            "I_S": [Decimal(0) for t in range(0, num_ticks + 1)],
            "I_A": [Decimal(0) for t in range(0, num_ticks + 1)],
            "R": [Decimal(0) for t in range(0, num_ticks + 1)],
        }
        self.history["S"][0] = self.population

        self.seasonality_factor = Decimal(1.0)

        # Temporary storage of flow that needs to be applied/removed
        self.symptomatic_flow = None
        self.asymptomatic_flow = None

    def set_flows(self, inflow, outflow):
        total_symptomatic_inflow = inflow[0]
        total_asymptomatic_inflow = inflow[1]

        total_symptomatic_outflow = outflow[0]
        total_asymptomatic_outflow = outflow[1]

        net_symptomatic_flow = total_symptomatic_inflow - total_symptomatic_outflow
        net_asymptomatic_flow = total_asymptomatic_inflow - total_asymptomatic_outflow

        self.symptomatic_flow = net_symptomatic_flow
        self.asymptomatic_flow = net_asymptomatic_flow

    def apply_flow(self, t):
        assert self.symptomatic_flow is not None and self.asymptomatic_flow is not None
        self.history["I_S"][t] += self.symptomatic_flow
        self.history["I_A"][t] += self.asymptomatic_flow

    def remove_flow(self, t):
        self.history["I_S"][t] -= self.symptomatic_flow
        self.history["I_A"][t] -= self.asymptomatic_flow
        self.symptomatic_flow = None
        self.asymptomatic_flow = None

    def update(self, t):
        b = CONSTANTS["beta"]
        R_b = CONSTANTS["rbeta"]
        e = CONSTANTS["epsilon"]
        pa = CONSTANTS["pa"]
        upa = CONSTANTS["upa"]
        u = CONSTANTS["mu"]

        self.apply_flow(t)

        S_t = Decimal(self.history["S"][t])
        E_t = Decimal(self.history["E"][t])
        I_S_t = Decimal(self.history["I_S"][t])
        I_A_t = Decimal(self.history["I_A"][t])
        R_t = Decimal(self.history["R"][t])

        NUM_TO_BATCH = 50

        ##########################
        # Susceptible --> Latent #
        ##########################
        new_latent = 0
        CONTACT_SYMPTOMATIC = (S_t / self.population) * (I_S_t / self.population)
        CONTACT_ASYMPTOMATIC = (S_t / self.population) * (I_A_t / self.population)

        if round(S_t) < NUM_TO_BATCH:
            for s in [rand() for i in range(round(S_t))]:
                a = (
                    CONTACT_ASYMPTOMATIC
                    * CONSTANTS["beta"]
                    * CONSTANTS["rbeta"]
                    * self.seasonality_factor
                )
                b = CONTACT_SYMPTOMATIC * CONSTANTS["beta"] * self.seasonality_factor
                smaller = min(a, b)
                larger = max(a, b)

                if s < smaller:
                    new_latent += 1
                elif s < smaller + larger:
                    new_latent += 1
        else:
            for s in [rand() for i in range(round(S_t) // NUM_TO_BATCH)]:
                a = (
                    CONTACT_ASYMPTOMATIC
                    * CONSTANTS["beta"]
                    * CONSTANTS["rbeta"]
                    * self.seasonality_factor
                )
                b = CONTACT_SYMPTOMATIC * CONSTANTS["beta"] * self.seasonality_factor
                smaller = min(a, b)
                larger = max(a, b)

                if s < smaller:
                    new_latent += NUM_TO_BATCH
                elif s < smaller + larger:
                    new_latent += NUM_TO_BATCH

        # new_latent = S_t / self.population * b * (I_A_t * R_b + I_S_t)

        self.remove_flow(t)

        #######################
        # Latent --> Infected #
        #######################
        new_infected_symptomatic = 0
        new_infected_asymptomatic = 0

        if round(E_t) < NUM_TO_BATCH:
            for e in [rand() for i in range(round(E_t))]:
                if e < CONSTANTS["epsilon"] * CONSTANTS["pa"]:  # .3
                    new_infected_asymptomatic += 1
                elif e < (
                    Decimal(0.3) + CONSTANTS["epsilon"] * CONSTANTS["upa"]
                ):  # .609
                    new_infected_symptomatic += 1
        else:
            for e in [rand() for i in range(round(E_t) // NUM_TO_BATCH)]:
                if e < CONSTANTS["epsilon"] * CONSTANTS["pa"]:  # .3
                    new_infected_asymptomatic += NUM_TO_BATCH
                elif e < Decimal(0.3) + CONSTANTS["epsilon"] * CONSTANTS["upa"]:  # .609
                    new_infected_symptomatic += NUM_TO_BATCH

        # new_infected_symptomatic = E_t * upt * upa
        # new_infected_asymptomatic = E_t * e * pa
        new_infected = new_infected_symptomatic + new_infected_asymptomatic

        ##########################
        # Infected --> Recovered #
        ##########################
        new_recovered_from_infected_symptomatic = 0
        new_recovered_from_infected_asymptomatic = 0
        if round(I_S_t) < NUM_TO_BATCH:
            for i_s in [rand() for i in range(round(I_S_t))]:
                if i_s < CONSTANTS["mu"]:
                    new_recovered_from_infected_symptomatic += 1
        else:
            for i_s in [rand() for i in range(round(I_S_t) // NUM_TO_BATCH)]:
                if i_s < CONSTANTS["mu"]:
                    new_recovered_from_infected_symptomatic += NUM_TO_BATCH

        if round(I_A_t) < NUM_TO_BATCH:
            for i_a in [rand() for i in range(round(I_A_t))]:
                if i_a < CONSTANTS["mu"]:
                    new_recovered_from_infected_asymptomatic += 1
        else:
            for i_a in [rand() for i in range(round(I_A_t) // NUM_TO_BATCH)]:
                if i_a < CONSTANTS["mu"]:
                    new_recovered_from_infected_asymptomatic += NUM_TO_BATCH

        # new_recovered_from_infected_symptomatic = I_S_t * u
        # new_recovered_from_infected_asymptomatic = I_A_t * u
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

        # Bounds checking
        if self.history["S"][t + 1] < 0:
            self.history["S"][t + 1] = 0
        if self.history["E"][t + 1] < 0:
            self.history["E"][t + 1] = 0
        if self.history["I_S"][t + 1] < 0:
            self.history["I_S"][t + 1] = 0
        if self.history["I_A"][t + 1] < 0:
            self.history["I_A"][t + 1] = 0
        if self.history["R"][t + 1] > self.population:
            self.history["R"][t + 1] = self.population

    def plot_history(self, ax, title):
        df = pd.DataFrame(self.history)
        make_compartment_charts(df, self.num_ticks, ax, title)

    def __str__(self):
        return pformat(self.history, indent=4)


class NTAGraphNode:
    def __init__(self, info, num_ticks):
        # Info looks like ('Borough', 'NTA ID', 'NTA Name', 'Population', 'Centroid Lat', 'Centroid Long', 'Hospitalization Rate')
        self.model = DiseaseModel(int(info[3]), num_ticks)
        # Tuple (lat, long)
        self.centroid = (float(info[4]), float(info[5]))
        self.info = info
        self.id = info[1]
        self.hospitalization_rate = float(info[6])

        self.s_movement_factor = Decimal(0.25)
        self.as_movement_factor = Decimal(0.75)
        self.distances_normalized = None
        self.movement_restriction = Decimal(1.0)

    def cache_distances(self, nodes):
        # This next section is called and computed after node creation to speed up the simulation later
        node_set = {k: v for k, v in nodes.items() if k != self.id}
        # Calculate the distance ratio for every other node
        distances = {n: self.distance_in_km(v.centroid) for n, v in node_set.items()}

        # for _ in range(10):
        #     max_distance_key = max(distances, key=lambda k: distances[k])
        #     del distances[max_distance_key]

        total_distance = sum(distances.values())
        self.distances_normalized = {
            n: d / total_distance for n, d in distances.items()
        }

    def seed(self, num):
        self.model.history["S"][0] -= Decimal(num)
        self.model.history["I_S"][0] += Decimal(num) / Decimal(2)
        self.model.history["I_A"][0] += Decimal(num) / Decimal(2)

    def distance_in_km(self, other_point):
        return Decimal(geodesic(self.centroid, other_point).meters) / Decimal(1000)

    def total_outflow(self, tick):
        return (
            Decimal(self.model.history["I_S"][tick]) * self.s_movement_factor,
            Decimal(self.model.history["I_A"][tick]) * self.as_movement_factor,
        )

    def flow(self, tick):
        # Flow from this node in the set to all other nodes
        I_S = Decimal(self.model.history["I_S"][tick]) * self.s_movement_factor
        I_A = Decimal(self.model.history["I_A"][tick]) * self.as_movement_factor

        max_distance = (
            max(self.distances_normalized.values()) * self.movement_restriction
        )

        linear_outflow = {
            n: (I_S * r, I_A * r) if r < max_distance else (0, 0)
            for n, r in self.distances_normalized.items()
        }
        # assert round(sum([flow[0] for flow in linear_outflow.values()])) == round(I_S)
        # assert round(sum([flow[1] for flow in linear_outflow.values()])) == round(I_A)
        # flow_diff = round(sum([flow[0] for flow in linear_outflow.values()])) - round(
        #     I_S
        # )
        # if flow_diff > 0:
        #     print("{} flow diff: {}".format(self.id, flow_diff))

        # nonlinear_distances = {n: 1 / d for n, d in distances_normalized.items()}
        # sum_nonlinear = sum(nonlinear_distances.values())
        # nonlinear_normalized = {
        #     n: d / sum_nonlinear for n, d in distances_normalized.items()
        # }

        # infected_outflow = {
        #     n: (I_S * r, I_A * r) for n, r in nonlinear_normalized.items()
        # }
        # print(infected_outflow)

        # print(sum([flow[0] for flow in infected_outflow.values()]), I_S)
        # print(sum([flow[1] for flow in infected_outflow.values()]), I_A)
        # assert int(sum([flow[0] for flow in infected_outflow.values()])) == I_S
        # assert int(sum([flow[1] for flow in infected_outflow.values()])) == I_A

        # return infected_outflow
        return linear_outflow

    def __str__(self):
        return "{} ({})".format(self.info[2], self.info[1])

    def __hash__(self):
        return hash(self.info[1])

    def __eq__(self, o):
        return hash(self.info[1]) == hash(o.info[1])

    def __repr__(self):
        return str(self.model)


class Simulation:
    def __init__(self, nta_data, hospital_data, num_ticks=100):
        self.num_ticks = num_ticks
        print("Generating NTA Nodes...")
        self.nodes = self.init_graph_nodes(nta_data)
        print("Caching Distances For Nodes...")
        [n.cache_distances(self.nodes) for n in self.nodes.values()]

    def init_graph_nodes(self, df):
        iterator = df.itertuples(index=False, name=None)
        next(iterator)  # skip the first line containing the headers
        nodes = {
            row[1]: NTAGraphNode(row, self.num_ticks) for row in iterator
        }  # Seed nodes
        # nodes["QN29"].seed(1000)
        # nodes["QN30"].seed(1000)
        # nodes["QN31"].seed(1000)
        nodes["MN25"].seed(1000)
        return nodes

    def run(self):
        # global RANDS

        # print("Generating Random Numbers...")
        # # total_pop = int(sum(n.model.population for n in self.nodes.values()))
        # # Total Population -- 6,699,917
        # total_rands = 6700000 * self.num_ticks
        # # Turn into dictionary for O(1) lookup
        # RANDS = dict(zip(np.arange(0, total_rands), np.random.rand(total_rands)))

        for i in range(self.num_ticks):
            print(
                "===============\nSimulating Day {}...\n===============".format(i + 1)
            )

            print("Calculating Flows Between All Nodes...")
            # { node_id: {other_node_id: flow-to-that-node, etc.} }
            flows = {node_id: node.flow(i) for node_id, node in self.nodes.items()}

            print("Setting Flows For Each Node Model...")
            for node_id, node in self.nodes.items():
                total_inflow = [Decimal(0), Decimal(0)]
                for flow_node_id, outflows in flows.items():
                    if node_id != flow_node_id:
                        total_inflow[0] += outflows[node_id][0]
                        total_inflow[1] += outflows[node_id][1]
                node.model.set_flows(total_inflow, node.total_outflow(i))

            print("Updating Each Node Model...")
            [node.model.update(i) for node in self.nodes.values()]

        aggregate_history = {
            "S": [0 for t in range(0, self.num_ticks + 1)],
            "E": [0 for t in range(0, self.num_ticks + 1)],
            "I_S": [0 for t in range(0, self.num_ticks + 1)],
            "I_A": [0 for t in range(0, self.num_ticks + 1)],
            "R": [0 for t in range(0, self.num_ticks + 1)],
        }

        for i in range(self.num_ticks + 1):
            for node in self.nodes.values():
                aggregate_history["S"][i] += node.model.history["S"][i]
                aggregate_history["E"][i] += node.model.history["E"][i]
                aggregate_history["I_S"][i] += node.model.history["I_S"][i]
                aggregate_history["I_A"][i] += node.model.history["I_A"][i]
                aggregate_history["R"][i] += node.model.history["R"][i]

        pprint(aggregate_history)
        # # Graph for all of New York
        fig, axs = plt.subplots(3)

        make_compartment_charts(
            pd.DataFrame(aggregate_history), self.num_ticks, axs[0], "All of NYC"
        )

        self.nodes["QN12"].model.plot_history(axs[1], "QN12 (other side of NYC)")
        pprint(self.nodes["QN12"].model.history)
        self.nodes["MN25"].model.plot_history(axs[2], "MN25 (seed)")
        pprint(self.nodes["MN25"].model.history)

        fig.suptitle("Compartment Population vs Time")
        plt.show()
        # self.nodes["QN30"].model.plot_history()
        # self.nodes["QN31"].model.plot_history()
        # self.nodes["QN54"].model.plot_history()
