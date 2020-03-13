# Comp-Epi-Project

## Project Goal

1. Make a contact network between people in NYC using NTA demographic data 
2. Model disease movement and the impact of being a densely populated urban area
3. Find tradeoffs between hospital accessiblity and social distancing movement restrictions

## Getting Started

```bash
$ pip install pipenv
$ pipenv install --dev
$ pipenv shell
$ python --version
Python 3.7.6
$ pip --version
pip 20.0.2 from /home/kevin/.local/share/virtualenvs/Comp-Epi-Project-EMfCM2jV/lib/python3.7/site-packages/pip (python 3.7)
$ python main.py
```

When VSCode asks to use an autoformatter, select `black`.

## Top Priorities
1. Construct updated model with described compartments in GLEAMviz
2. Map out hospitals in correlation to NTA locations 
3. Assign hospitals bed values and fill in missing Lat/Long coordinates
4. Establish node characteristics and optimizing hospital priorization

## Data Sources

[1](https://data.cityofnewyork.us/City-Government/Neighborhood-Tabulation-Areas-NTA-/cpf4-rkhq)  
[2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0016591)  
[3](https://data.cityofnewyork.us/City-Government/Census-Demographics-at-the-Neighborhood-Tabulation/rnsn-acs2)  
[4](https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_ID_46)  
[5](https://data.cityofnewyork.us/Health/NYC-Health-Hospitals-Facilities-2011/ymhw-9cz9)  
[6](https://profiles.health.ny.gov/hospital/bed_type/Intensive+Care+Beds)  
[7](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3034199/)  
[8](https://public.tableau.com/profile/nyc.health#!/vizhome/NewYorkCityNeighborhoodHealthAtlas/Home)  
[9](https://www.ncbi.nlm.nih.gov/books/NBK126700/)  



