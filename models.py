from geopy.distance import geodesic
import matplotlib.pyplot as plt
import pandas as pd

from pprint import pprint, pformat
from decimal import *


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


def make_compartment_charts(df, num_ticks):
    t = range(0, num_ticks + 1)
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

        # Net flow might be causing the bugs here
        # print(
        #     """
        # Total I_S Inflow: {}\n
        # Total I_A Inflow: {}\n
        # Total I_S Outflow: {}\n
        # Total I_A Outflow: {}\n
        # Net I_S: {}\n
        # Net I_A: {}\n
        # """.format(
        #         total_symptomatic_inflow,
        #         total_asymptomatic_inflow,
        #         total_symptomatic_outflow,
        #         total_asymptomatic_outflow,
        #         net_symptomatic_flow,
        #         net_asymptomatic_flow,
        #     )
        # )

        # self.symptomatic_flow = net_symptomatic_flow
        # self.asymptomatic_flow = net_asymptomatic_flow
        # assert total_symptomatic_inflow >= 0
        # assert total_asymptomatic_inflow >= 0
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
        # assert (
        #     self.history["S"][t]
        #     + self.history["E"][t]
        #     + self.history["I_S"][t]
        #     + self.history["I_A"][t]
        #     + self.history["R"][t]
        #     == self.population
        # )

        b = CONSTANTS["beta"]
        R_b = CONSTANTS["rbeta"]
        e = CONSTANTS["epsilon"]
        pa = CONSTANTS["pa"]
        upt = CONSTANTS["upt"]
        upa = CONSTANTS["upa"]
        u = CONSTANTS["mu"]

        self.apply_flow(t)

        S_t = Decimal(self.history["S"][t])
        E_t = Decimal(self.history["E"][t])
        I_S_t = Decimal(self.history["I_S"][t])
        I_A_t = Decimal(self.history["I_A"][t])
        R_t = Decimal(self.history["R"][t])

        assert S_t <= self.population and S_t >= 0

        new_latent = S_t / self.population * b * (I_A_t * R_b + I_S_t)

        # new_latent_from_travel = S_t / self.population * b * (self.asymptomatic_flow * R_b + self.symptomatic_flow)

        new_infected_symptomatic = E_t * upt * upa
        new_infected_asymptomatic = E_t * e * pa
        new_infected = new_infected_symptomatic + new_infected_asymptomatic
        new_recovered_from_infected_symptomatic = I_S_t * u
        new_recovered_from_infected_asymptomatic = I_A_t * u
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

        ###################
        # Bounds checking #
        ###################
        if self.history["S"][t + 1] < 0:
            self.history["S"][t + 1] = 0
        if self.history["E"][t + 1] < 0:
            self.history["E"][t + 1] = 0
        if self.history["I_S"][t + 1] < 0:
            self.history["I_S"][t + 1] = 0
        if self.history["I_A"][t + 1] < 0:
            self.history["I_A"][t + 1] = 0
        if self.history["R"][t + 1] < 0:
            # raise ValueError("The recovered compartment should not be less than 0.")
            print("The recovered compartment should not be less than 0.")

        # assert (
        #     self.history["S"][t + 1]
        #     + self.history["E"][t + 1]
        #     + self.history["I_S"][t + 1]
        #     + self.history["I_A"][t + 1]
        #     + self.history["R"][t + 1]
        #     == self.population
        # )

        print(
            # self.history["S"][t + 1],
            # self.history["E"][t + 1],
            # self.history["I_S"][t + 1],
            # self.history["I_A"][t + 1],
            # self.history["R"][t + 1],
            (
                self.history["S"][t + 1]
                - self.history["S"][t]
                + self.history["E"][t + 1]
                - self.history["E"][t]
                + self.history["I_S"][t + 1]
                - self.history["I_S"][t]
                + self.history["I_A"][t + 1]
                - self.history["I_A"][t]
                + self.history["R"][t + 1]
                - self.history["R"][t]
            )
        )

    def plot_history(self):
        df = pd.DataFrame(self.history)
        make_compartment_charts(df, self.num_ticks)

    def __str__(self):
        return pformat(self.history, indent=4)


class NTAGraphNode:
    def __init__(self, info, num_ticks, seed_num=None):
        # Info looks like ('Borough', 'NTA ID', 'NTA Name', 'Population', 'Centroid Lat', 'Centroid Long', 'Hospitalization Rate')
        self.model = DiseaseModel(int(info[3]), num_ticks)
        # Tuple (lat, long)
        self.centroid = (float(info[4]), float(info[5]))
        self.info = info
        self.id = info[1]
        self.hospitalization_rate = float(info[6])
        # 50% of the infected compartments move
        self.movement_factor = Decimal(0.5)

        if seed_num:
            self.seed(seed_num)

    def seed(self, num):
        self.model.history["S"][0] -= Decimal(num)
        self.model.history["I_S"][0] += Decimal(num) / Decimal(2)
        self.model.history["I_A"][0] += Decimal(num) / Decimal(2)

    def distance_in_km(self, other_point):
        return Decimal(geodesic(self.centroid, other_point).meters) / Decimal(1000)

    # def daily_flow(self, other_node, tick):
    #     # Assume 50% of the inhabitants move to another NTA during the day and come back at night
    #     # This essentially means, assuming uniform movement like we have been (e.g. infected people)
    #     # do not curtail their travel, that 50% of the I_A and I_S compartments in this NTA should
    #     # apply to the other NTAs. We will basically "add" them to the compartment before the update
    #     # calculation, then remove them afterwards. This will be weighted by the distance between
    #     # the source and destination node, so that there is 50% of the movement to NTAs 5km away.
    #     movement_factor = 0.5

    #     distance = self.distance_in_km(other_node.centroid)
    #     distance_factor = 1 / (1 + (distance / 5) ** 2)

    #     inf_symptomatic = int(
    #         self.model.history["I_S"][tick] * distance_factor * movement_factor
    #     )
    #     inf_asymptomatic = int(
    #         self.model.history["I_A"][tick] * distance_factor * movement_factor
    #     )

    #     return (inf_symptomatic, inf_asymptomatic)

    def total_outflow(self, tick):
        return (
            Decimal(self.model.history["I_S"][tick]) * self.movement_factor,
            Decimal(self.model.history["I_A"][tick]) * self.movement_factor,
        )

    def flow(self, node_set, tick):
        # Flow from this node in the set to all other nodes
        I_S = Decimal(self.model.history["I_S"][tick]) * self.movement_factor
        I_A = Decimal(self.model.history["I_A"][tick]) * self.movement_factor

        # Calculate the distance ratio for every other node
        distances = {n: self.distance_in_km(v.centroid) for n, v in node_set.items()}

        # for _ in range(10):
        #     max_distance_key = max(distances, key=lambda k: distances[k])
        #     del distances[max_distance_key]

        # pprint(distances)
        total_distance = sum(distances.values())
        distances_normalized = {n: d / total_distance for n, d in distances.items()}

        linear_outflow = {
            n: (I_S * r, I_A * r) for n, r in distances_normalized.items()
        }
        assert round(sum([flow[0] for flow in linear_outflow.values()])) == round(I_S)
        assert round(sum([flow[1] for flow in linear_outflow.values()])) == round(I_A)

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
        self.nodes = self.init_graph_nodes(nta_data)

    def init_graph_nodes(self, df):
        iterator = df.itertuples(index=False, name=None)
        next(iterator)  # skip the first line containing the headers
        first_node = NTAGraphNode(next(iterator), self.num_ticks, seed_num=1000)
        nodes = {first_node.id: first_node}
        nodes.update({row[1]: NTAGraphNode(row, self.num_ticks) for row in iterator})
        return nodes

    def run(self):
        for i in range(self.num_ticks):
            print(
                "===============\nSimulating Day {}...\n===============".format(i + 1)
            )

            print("Calculating Flows Between All Nodes...")
            # { node_id: {other_node_id: flow-to-that-node, etc.} }
            flows = {
                node_id: node.flow(
                    {k: v for k, v in self.nodes.items() if k != node_id}, i
                )
                for node_id, node in self.nodes.items()
            }

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

        # aggregate_history = {
        #     "S": [0 for t in range(0, self.num_ticks + 1)],
        #     "E": [0 for t in range(0, self.num_ticks + 1)],
        #     "I_S": [0 for t in range(0, self.num_ticks + 1)],
        #     "I_A": [0 for t in range(0, self.num_ticks + 1)],
        #     "R": [0 for t in range(0, self.num_ticks + 1)],
        # }

        # for i in range(self.num_ticks + 1):
        #     for node in self.nodes:
        #         aggregate_history["S"][i] += node.model.history["S"][i]
        #         aggregate_history["E"][i] += node.model.history["E"][i]
        #         aggregate_history["I_S"][i] += node.model.history["I_S"][i]
        #         aggregate_history["I_A"][i] += node.model.history["I_A"][i]
        #         aggregate_history["R"][i] += node.model.history["R"][i]

        # print(aggregate_history)
        # Graph for all of New York
        # make_compartment_charts(pd.DataFrame(aggregate_history), self.num_ticks)

        # distances = []
        # for n in self.nodes:
        #     for k in self.nodes:
        #         if n != k:
        #             distances.append(n.distance_in_meters(k.centroid))
        #             print(
        #                 "[{} -> {}]\t\t\t".format(n, k),
        #                 n.distance_in_meters(k.centroid),
        #             )
        # print(min(distances))  # 0.20199763118451938
        # print(max(distances))  # 59.657808168182136
