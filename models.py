from geopy.distance import geodesic
import matplotlib.pyplot as plt
import pandas as pd

from pprint import pprint, pformat
from numpy.random import rand, randint
from random import shuffle
from enum import Enum
import json
import multiprocessing


# Number of days to run the simulation for
NUM_TICKS = 100
BOUND = NUM_TICKS
NUM_TO_BATCH = 10
TERTIARY_CARE_THRESHOLD = 60  # 62, 85, 85, 99, 161 are the highest bed counts

AVG_LATENCY_PERIOD = 5.1
AVG_INFECTIOUS_PERIOD = 13.25
# AVG_LATENCY_PERIOD = 1.1
# AVG_INFECTIOUS_PERIOD = 2.5
CONSTANTS = {
    "beta": 0.8383,
    "epsilon": 1 / AVG_LATENCY_PERIOD,
    "mu": 1 / AVG_INFECTIOUS_PERIOD,
    "pa": 0.33,
    "pt": 0.5,
    "rbeta": 0.5,
    "upa": 1 - 0.33,
    "upt": 1 - 0.5,
}

# Slider for the hospital distance radius, social distancing factor, and travel radius
MOVEMENT_RESTRICTION = 1.0
# MOVEMENT_RESTRICTION = 0.5

# The proportion of the individuals in an NTA will actually move to other NTAs during a given day
BASE_MOVEMENT_FACTOR = 0.4
I_S_MOVEMENT_FACTOR = BASE_MOVEMENT_FACTOR * 0.5
I_A_MOVEMENT_FACTOR = BASE_MOVEMENT_FACTOR

# Parameter scaling transmissibility depending on the season we wish to model
SEASONALITY_FACTOR = 1.0

# Percent asymptomatic carriers (when seeding)
PERCENT_ASYMPTOMATIC = 0.5

# Temp variable to estimate hospitalizations
TOTAL_HOSPITALIZED = 0
TOTAL_DEAD = 0


class Compartment(Enum):
    susceptible = "S"
    latent = "E"
    infectious_symptomatic = "I_S"
    infectious_asymptomatic = "I_A"
    recovered = "R"
    needs_hospitalization = "NH"
    hospitalized = "H"
    dead = "D"

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()


def worker(arg):
    node, c_s, c_a = arg
    node.transition(c_s, c_a)


def make_compartment_charts(df, ax, title):
    t = range(0, NUM_TICKS)
    ax.plot(
        t,
        df[Compartment.susceptible],
        markerfacecolor="blue",
        markersize=8,
        color="skyblue",
        linewidth=2,
        label="Susceptible",
    )
    ax.plot(
        t,
        df[Compartment.latent],
        markerfacecolor="purple",
        markersize=8,
        color="purple",
        linewidth=2,
        label="Latent",
    )
    ax.plot(
        t,
        df[Compartment.infectious_symptomatic],
        markerfacecolor="red",
        markersize=8,
        color="red",
        linewidth=2,
        label="Infectious (Symptomatic)",
    )
    ax.plot(
        t,
        df[Compartment.infectious_asymptomatic],
        markerfacecolor="orange",
        markersize=8,
        color="orange",
        linewidth=2,
        label="Infectious (Asymptomatic)",
    )
    ax.plot(
        t,
        df[Compartment.recovered],
        markerfacecolor="green",
        markersize=8,
        color="green",
        linewidth=2,
        label="Recovered",
    )
    # plt.xlabel("Day")
    # plt.ylabel("# in Compartment")
    # ax.legend()
    ax.set_title(title, fontdict={"fontsize": 16, "fontweight": "medium"})


class Individual:
    def __init__(self, num, nta_id, size, hospitalization_rate):
        self.compartment = Compartment.susceptible
        self.id = num
        self.home_nta = nta_id
        self.size = size
        self.hospitalization_rate = hospitalization_rate
        self.round_num = 5
        self.multiplier = 10 ** self.round_num

    def transition(self, CONTACT_SYMPTOMATIC, CONTACT_ASYMPTOMATIC):
        global TOTAL_DEAD

        if not (
            self.compartment == Compartment.dead
            or self.compartment == Compartment.recovered
        ):
            if self.compartment == Compartment.susceptible:
                self.infect(CONTACT_ASYMPTOMATIC, CONTACT_SYMPTOMATIC)
            elif self.compartment == Compartment.latent:
                self.make_infectious()
            elif (
                self.compartment == Compartment.infectious_symptomatic
                or self.compartment == Compartment.infectious_asymptomatic
            ):
                self.recover()
            # elif self.compartment == Compartment.needs_hospitalization:
            #     roll = randint(0, 10000)
            # if roll < 500:
            #     self.compartment = Compartment.dead
            #     TOTAL_DEAD += NUM_TO_BATCH
            # elif roll < 2500:
            #     self.compartment = Compartment.recovered
            # if roll < 2500:
            #     self.compartment = Compartment.recovered

    def infect(self, CONTACT_SYMPTOMATIC, CONTACT_ASYMPTOMATIC):
        threshold1 = (
            round(
                CONTACT_ASYMPTOMATIC
                * CONSTANTS["beta"]
                * CONSTANTS["rbeta"]
                * SEASONALITY_FACTOR,
                self.round_num,
            )
            * self.multiplier
        )
        threshold2 = (
            round(
                CONTACT_SYMPTOMATIC * CONSTANTS["beta"] * SEASONALITY_FACTOR,
                self.round_num,
            )
            * self.multiplier
            + threshold1
        )
        roll = randint(0, self.multiplier)

        if roll < threshold1:
            self.compartment = Compartment.latent
        elif roll < threshold2:
            self.compartment = Compartment.latent

    def make_infectious(self):
        threshold1 = (
            round(CONSTANTS["epsilon"] * CONSTANTS["pa"], self.round_num,)
            * self.multiplier
        )
        threshold2 = (
            round(CONSTANTS["epsilon"] * CONSTANTS["upa"], self.round_num,)
            * self.multiplier
        ) + threshold1
        roll = randint(0, self.multiplier)

        if roll < threshold1:
            self.compartment = Compartment.infectious_asymptomatic
        elif roll < threshold2:
            self.compartment = Compartment.infectious_symptomatic
            global TOTAL_HOSPITALIZED
            # critical_threshold = round(self.hospitalization_rate, 8) * (10 ** 8)
            critical_threshold = round(0.2655, 8) * (10 ** 8)
            if randint(0, (10 ** 8)) < critical_threshold:
                # self.compartment = Compartment.needs_hospitalization
                TOTAL_HOSPITALIZED += NUM_TO_BATCH

    def recover(self):
        threshold = round(CONSTANTS["mu"], self.round_num) * self.multiplier
        roll = randint(0, self.multiplier)

        if roll < threshold:
            self.compartment = Compartment.recovered

    def kill(self):
        # TODO
        pass

    def __str__(self):
        return str(self.home_nta)

    def __repr__(self):
        return self.__str__()

    # def __hash__(self):
    #     return hash((self.id, self.home_nta))

    # def __eq__(self, o):
    #     if not isinstance(o, type(self)): return NotImplemented
    #     return self.id == o.id and self.home_nta == o.home_nta


class DiseaseModel:
    def __init__(self, population, nta_id, hospitalization_rate):
        self.overflow = population % NUM_TO_BATCH
        self.population = population
        if self.overflow != 0:
            self.population += NUM_TO_BATCH - self.overflow
        self.individuals = [
            Individual(i, nta_id, NUM_TO_BATCH, hospitalization_rate)
            for i in range(0, self.population + 1, NUM_TO_BATCH)
        ]
        self.history = {
            Compartment.susceptible: [0 for t in range(0, BOUND)],
            Compartment.latent: [0 for t in range(0, BOUND)],
            Compartment.infectious_symptomatic: [0 for t in range(0, BOUND)],
            Compartment.infectious_asymptomatic: [0 for t in range(0, BOUND)],
            Compartment.recovered: [0 for t in range(0, BOUND)],
            Compartment.needs_hospitalization: [0 for t in range(0, BOUND)],
            Compartment.hospitalized: [0 for t in range(0, BOUND)],
            Compartment.dead: [0 for t in range(0, BOUND)],
        }

    def count_compartments(self):
        s = 0
        e = 0
        i_s = 0
        i_a = 0
        r = 0
        d = 0
        for i in self.individuals:
            if i.compartment == Compartment.susceptible:
                s += 1
            if i.compartment == Compartment.latent:
                e += 1
            if i.compartment == Compartment.infectious_symptomatic:
                i_s += 1
            if i.compartment == Compartment.infectious_asymptomatic:
                i_a += 1
            if i.compartment == Compartment.recovered:
                r += 1
            if i.compartment == Compartment.dead:
                d += 1
        return (s, e, i_s, i_a, r, d)

    def count_compartment_pop(self):
        s = 0
        e = 0
        i_s = 0
        i_a = 0
        r = 0
        nh = 0
        h = 0
        d = 0
        for i in self.individuals:
            if i.compartment == Compartment.susceptible:
                s += i.size
            if i.compartment == Compartment.latent:
                e += i.size
            if i.compartment == Compartment.infectious_symptomatic:
                i_s += i.size
            if i.compartment == Compartment.infectious_asymptomatic:
                i_a += i.size
            if i.compartment == Compartment.recovered:
                r += i.size
            if i.compartment == Compartment.needs_hospitalization:
                nh += i.size
            if i.compartment == Compartment.hospitalized:
                h += i.size
            if i.compartment == Compartment.dead:
                d += i.size
        return (s, e, i_s, i_a, r, nh, h, d)

    def update_history(self, t):
        s, e, i_s, i_a, r, nh, h, d = self.count_compartment_pop()
        self.history[Compartment.susceptible][t] = s
        self.history[Compartment.latent][t] = e
        self.history[Compartment.infectious_symptomatic][t] = i_s
        self.history[Compartment.infectious_asymptomatic][t] = i_a
        self.history[Compartment.recovered][t] = r
        self.history[Compartment.needs_hospitalization][t] = nh
        self.history[Compartment.hospitalized][t] = h
        self.history[Compartment.dead][t] = d

    def update(self):
        s, e, i_s, i_a, r, nh, h, d = self.count_compartment_pop()
        N = self.population
        CONTACT_SYMPTOMATIC = (s / N) * ((i_s + nh + h) / N) * 2
        CONTACT_ASYMPTOMATIC = (s / N) * (i_a / N) * 2

        # try:
        #     cpus = multiprocessing.cpu_count()
        # except NotImplementedError:
        #     cpus = 1
        # with multiprocessing.Pool(processes=cpus) as p:
        #     p.map(
        #         worker,
        #         (
        #             (i, CONTACT_SYMPTOMATIC, CONTACT_ASYMPTOMATIC)
        #             for i in self.individuals
        #         ),
        #     )
        [
            i.transition(CONTACT_SYMPTOMATIC, CONTACT_ASYMPTOMATIC)
            for i in self.individuals
        ]

    def seed(self, num):
        asymptomatic = int(num / NUM_TO_BATCH * PERCENT_ASYMPTOMATIC)
        for i in self.individuals[:asymptomatic]:
            i.compartment = Compartment.infectious_asymptomatic
        for i in self.individuals[asymptomatic : int(num / NUM_TO_BATCH)]:
            i.compartment = Compartment.infectious_symptomatic

    def plot_history(self, ax, title):
        df = pd.DataFrame(self.history)
        make_compartment_charts(df, ax, title)

    def __str__(self):
        return pformat(self.history, indent=4)


class HospitalNode:
    def __init__(self, info):
        self.name = info[0]
        self.centroid = (float(info[2]), float(info[1]))
        self.bed_count = int(info[3])

        if self.bed_count > TERTIARY_CARE_THRESHOLD:
            self.tertiary = True
        else:
            self.tertiary = False

    def __str__(self):
        return str("{} - {}".format(self.name, self.bed_count))

    def __repr__(self):
        return self.__str__()

    def distance_in_km(self, other_point):
        return geodesic(self.centroid, other_point).meters / 1000


class NTAGraphNode:
    def __init__(self, info):
        # Info looks like ('Borough', 'NTA ID', 'NTA Name', 'Population', 'Centroid Lat', 'Centroid Long', 'Hospitalization Rate')
        self.model = DiseaseModel(int(info[3]), info[1], float(info[6]))
        # Tuple (lat, long)
        self.centroid = (float(info[4]), float(info[5]))
        self.info = info
        self.id = info[1]
        self.hospitalization_rate = float(info[6])

        self.hospital_distances = None
        self.distances_normalized = None

    def cache_distances(self, nodes, hospitals):
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

        # Calculating distance to each hospital
        self.hospital_distances = {
            h.name: round(self.distance_in_km(h.centroid), 4)
            for h in hospitals
            if not h.tertiary
        }

    def distance_in_km(self, other_point):
        return geodesic(self.centroid, other_point).meters / 1000

    def calculate_outflow(self):
        s, e, i_s, i_a, r, _, _, _ = self.model.count_compartment_pop()

        s *= BASE_MOVEMENT_FACTOR
        e *= BASE_MOVEMENT_FACTOR
        i_s *= I_S_MOVEMENT_FACTOR
        i_a *= BASE_MOVEMENT_FACTOR
        r *= BASE_MOVEMENT_FACTOR

        max_distance = max(self.distances_normalized.values()) * MOVEMENT_RESTRICTION
        # total_ds = len(self.distances_normalized.values())
        # zeroed = len(
        #     [d for d in self.distances_normalized.values() if d >= max_distance]
        # )
        # print(f"{zeroed} out of {total_ds} NTA destinations have been zeroed")

        linear_outflow = {
            n: {
                Compartment.susceptible: round(s * d / NUM_TO_BATCH),
                Compartment.latent: round(e * d / NUM_TO_BATCH),
                Compartment.infectious_symptomatic: round(i_s * d / NUM_TO_BATCH),
                Compartment.infectious_asymptomatic: round(i_a * d / NUM_TO_BATCH),
                Compartment.recovered: round(r * d / NUM_TO_BATCH),
            }
            if d < max_distance
            else {
                Compartment.susceptible: 0,
                Compartment.latent: 0,
                Compartment.infectious_symptomatic: 0,
                Compartment.infectious_asymptomatic: 0,
                Compartment.recovered: 0,
            }
            for n, d in self.distances_normalized.items()
        }
        return linear_outflow

    def find_and_pop(self, compartment):
        val = None
        for i, o in enumerate(self.model.individuals):
            if o.compartment == compartment:
                val = self.model.individuals.pop(i)
                break
        return val

    def __str__(self):
        return "{} ({})".format(self.info[2], self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, o):
        return hash(self.id) == hash(o.id)

    def __repr__(self):
        return str(self.model)


class Simulation:
    def __init__(self, nta_data, hospital_data):
        self.nodes = self.init_graph_nodes(nta_data)
        self.hospitals = self.init_hospital_nodes(hospital_data)
        [n.cache_distances(self.nodes, self.hospitals) for n in self.nodes.values()]

    def init_graph_nodes(self, df):
        iterator = df.itertuples(index=False, name=None)
        nodes = {row[1]: NTAGraphNode(row) for row in iterator}

        # Seed nodes
        # Central location
        nodes["QN19"].model.seed(1000)
        nodes["QN20"].model.seed(1000)
        nodes["QN21"].model.seed(1000)
        nodes["QN30"].model.seed(1000)
        nodes["BK77"].model.seed(1000)

        return nodes

    def init_hospital_nodes(self, df):
        iterator = df.itertuples(index=False, name=None)
        nodes = [HospitalNode(row) for row in iterator]
        return nodes

    def aggregate_history(self):
        h = {
            Compartment.susceptible: [0 for t in range(0, BOUND)],
            Compartment.latent: [0 for t in range(0, BOUND)],
            Compartment.infectious_symptomatic: [0 for t in range(0, BOUND)],
            Compartment.infectious_asymptomatic: [0 for t in range(0, BOUND)],
            Compartment.recovered: [0 for t in range(0, BOUND)],
            Compartment.dead: [0 for t in range(0, BOUND)],
        }
        for i in range(BOUND):
            for n in self.nodes.values():
                h[Compartment.susceptible][i] += n.model.history[
                    Compartment.susceptible
                ][i]
                h[Compartment.latent][i] += n.model.history[Compartment.latent][i]
                h[Compartment.infectious_symptomatic][i] += n.model.history[
                    Compartment.infectious_symptomatic
                ][i]
                h[Compartment.infectious_asymptomatic][i] += n.model.history[
                    Compartment.infectious_asymptomatic
                ][i]
                h[Compartment.recovered][i] += n.model.history[Compartment.recovered][i]
                h[Compartment.dead][i] += n.model.history[Compartment.dead][i]
        return h

    def write_out_history(self):
        with open(
            f"simulation-results_COVID_{NUM_TICKS}_{NUM_TO_BATCH}_{int(MOVEMENT_RESTRICTION*100)}%.csv",
            "w",
        ) as f:
            for n in self.nodes.values():
                cleaned = {}
                cleaned["S"] = [
                    round(v) for v in n.model.history[Compartment.susceptible]
                ]
                cleaned["E"] = [round(v) for v in n.model.history[Compartment.latent]]
                cleaned["I_S"] = [
                    round(v)
                    for v in n.model.history[Compartment.infectious_symptomatic]
                ]
                cleaned["I_A"] = [
                    round(v)
                    for v in n.model.history[Compartment.infectious_asymptomatic]
                ]
                cleaned["R"] = [
                    round(v) for v in n.model.history[Compartment.recovered]
                ]
                f.write("{}|{}\n".format(n.id, cleaned))

    def run(self, movement_restriction):
        global MOVEMENT_RESTRICTION
        MOVEMENT_RESTRICTION = movement_restriction

        for tick in range(NUM_TICKS):
            print(
                "===============\nSimulating Day {}...\n===============".format(
                    tick + 1
                )
            )

            print("Calculating Flows Between All Nodes...")
            # { node_id: {other_node_id: {S: flow-s, I_S: flow-I_S, ... } } }
            flows = {
                node_id: node.calculate_outflow()
                for node_id, node in self.nodes.items()
            }

            print("Moving Flows Between All Nodes...")
            for origin_node_id, outflows in flows.items():
                for dest_node_id, flow_to_that_node in outflows.items():
                    for c in [
                        Compartment.susceptible,
                        Compartment.latent,
                        Compartment.infectious_symptomatic,
                        Compartment.infectious_asymptomatic,
                        Compartment.recovered,
                    ]:
                        for _ in range(flow_to_that_node[c]):
                            val = self.nodes[origin_node_id].find_and_pop(c)
                            if val is not None:
                                self.nodes[dest_node_id].model.individuals.append(val)

            print("Updating All Node Models...")
            [node.model.update() for node in self.nodes.values()]

            # Move flows back at the end of each day
            print("Moving Flows Back To Home NTAs...")
            for n_id in self.nodes:
                # Build set of indices of nodes that don't belong
                indices = {
                    i
                    for (i, o) in enumerate(self.nodes[n_id].model.individuals)
                    if o.home_nta != n_id
                }
                # List of individuals that belong to this NTA
                home_indivs = [
                    value
                    for (i, value) in enumerate(self.nodes[n_id].model.individuals)
                    if i not in indices
                ]
                # Send others back to their original NTAs
                [
                    self.nodes[value.home_nta].model.individuals.append(value)
                    for (i, value) in enumerate(self.nodes[n_id].model.individuals)
                    if i in indices
                ]
                # Change list of origin individuals
                self.nodes[n_id].model.individuals = home_indivs

            [node.model.update_history(tick) for node in self.nodes.values()]

        total_nyc = self.aggregate_history()
        fig, axs = plt.subplots(3, 1, figsize=(16, 16))
        self.nodes["QN20"].model.plot_history(
            axs[0], "QN20 - Ridgewood (seed, Queens)"
        )  # Ridgewood (seed)
        self.nodes["SI07"].model.plot_history(
            axs[1], "SI07 - Westerleigh (low pop, Staten Island)"
        )  # Arden Heights (low pop)
        self.nodes["BK88"].model.plot_history(
            axs[2], "BK88 - Borough Park (high pop, Brooklyn)"
        )  # Borough Park (high pop)

        fig = plt.gcf()
        plt.savefig(
            f"results/specific_COVID_days_{NUM_TICKS}-batch_{NUM_TO_BATCH}-restr_{int(MOVEMENT_RESTRICTION*100)}%.png",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()

        global TOTAL_HOSPITALIZED

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        make_compartment_charts(pd.DataFrame(total_nyc), axs, "All of NYC")
        fig = plt.gcf()
        plt.savefig(
            f"results/totalnyc_COVID_days_{NUM_TICKS}-batch_{NUM_TO_BATCH}-restr_{int(MOVEMENT_RESTRICTION*100)}%-hosp_{TOTAL_HOSPITALIZED}.png",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()

        self.write_out_history()
