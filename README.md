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

## Missing elements to expand on in the future
1. More granular data points to model movement, specifically we need to see how people move during the day and what areas have higher population density at given times as that will effect when people move into hospitals.
2. Implementing hospitals in a more meaningful way, currently they get unrealisitcally overwhelmed quickly as individual choices to enter a hospital are taken as batched rolls.

## Data Sources

[1. New York City NTA GIS Data](https://data.cityofnewyork.us/City-Government/Neighborhood-Tabulation-Areas-NTA-/cpf4-rkhq)  
[2. Human Mobility Networks, Travel Restrictions, and the Global Spread of 2009 H1N1 Pandemic](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0016591)  
[3. Census Demographics at the NTA level](https://data.cityofnewyork.us/City-Government/Census-Demographics-at-the-Neighborhood-Tabulation/rnsn-acs2)  
[4. NYC and Tokyo FourSquare Check-in Dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_ID_46)  
[5. NYC Health and Hospital Facilities Location Data](https://data.cityofnewyork.us/Health/NYC-Health-Hospitals-Facilities-2011/ymhw-9cz9)  
[6. New York Hospital Bed Counts](https://profiles.health.ny.gov/hospital/bed_type/Intensive+Care+Beds)  
[7. The infection attack rate and severity of 2009 pandemic influenza (H1N1) in Hong Kong](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3034199/)  
[8. NYC NTA Age Bracket Breakdown](https://public.tableau.com/profile/nyc.health#!/vizhome/NewYorkCityNeighborhoodHealthAtlas/Home)  
[9. Emergency Department Visits and Hospital Inpatient Stays for Seasonal and 2009 H1N1 Influenza, 2008–2009](https://www.ncbi.nlm.nih.gov/books/NBK126700/)  



