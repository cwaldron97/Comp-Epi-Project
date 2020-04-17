from geopy.distance import geodesic

CONSTANTS = {
    "a_min": 0.7,
    "beta": 0.8383,
    "epsilon": 1 / 1.1,
    "mu": 1 / 2.5,
    "pa": 0.33,
    "pt": 0.5,
    "rbeta": 0.5,
    "upa": 1 - 0.33,
    "upt": 1 - 0.5,
}


class DiseaseModel:
    def __init__(self, population):
        self.population = population
        # time resolution is 1 day
        self.susceptible = None  # not infected
        self.latent = None  # infected but not yet infectious
        self.infectious_symptomatic = None
        self.infectious_asymptomatic = None
        self.recovered = None

    def susceptible_to_latent_via_symptomatic_contact(self):
        return CONSTANTS["beta"]

    def susceptible_to_latent_via_asymptomatic_contact(self):
        return CONSTANTS["beta"] * CONSTANTS["rbeta"]

    def latent_to_infectious_symptomatic(self):
        return CONSTANTS["epsilon"] * CONSTANTS["upt"] * CONSTANTS["upa"]

    def latent_to_infectious_asymptomatic(self):
        return CONSTANTS["epsilon"] * CONSTANTS["pa"]

    def infectious_to_recovered(self):
        return CONSTANTS["mu"]


class NTAGraphNode:
    def __init__(self, population, centroid):
        # DiseaseModel
        self.model = DiseaseModel(population)
        # Tuple (lat, long)
        self.centroid = centroid
        # List of neighboring nodes
        self.neighbors = []

    def distance_in_meters(self, other_point):
        return geodesic(self.center, other_point).meters
