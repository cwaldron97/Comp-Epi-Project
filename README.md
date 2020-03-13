# Comp-Epi-Project

## Project Goal

1. Make a contact network between people in Tokyo and NYC using check-in data 
2. Model disease movement and the impact of being a densely populated urban area
3. Find novel factors influencing disease spread

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

1. Picking what disease we want to study, and finding a way to model its infectivity and how it might propagate in an urban environment
2. Finding supplementary information to differentiate the cities, like transportation links (e.g. to adapt our distance function -- if someone can take a subway they're probably close to people on the other side of the city by that subway, spatially-temporally), access to health services, availability of public transportation, and differences in cultural practices (e.g. notions of personal space potentially impacting average distances people hold from one another)

## Potential Data Sources

[1](https://data.cityofnewyork.us/City-Government/Neighborhood-Tabulation-Areas-NTA-/cpf4-rkhq)  
[2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0016591)  
[3](https://data.cityofnewyork.us/City-Government/Census-Demographics-at-the-Neighborhood-Tabulation/rnsn-acs2)  
[4](https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_ID_46)  
[5](https://data.cityofnewyork.us/Health/NYC-Health-Hospitals-Facilities-2011/ymhw-9cz9)  
[6](https://profiles.health.ny.gov/hospital/bed_type/Intensive+Care+Beds)  
[7](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3034199/)  
[8](https://public.tableau.com/profile/nyc.health#!/vizhome/NewYorkCityNeighborhoodHealthAtlas/Home)  
[9](https://www.ncbi.nlm.nih.gov/books/NBK126700/)  



## Cities for research and writeup

[Modeling User Activity Preference by Leveraging User Spatial Temporal Characteristics in LBSNs](http://www-public.imtbs-tsp.eu/~zhang_da/pub/TSMC_YANG_2014.pdf)  
[How urbanization affects the epidemiology of emerging infectious diseases](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4481042/?fbclid=IwAR15JlnpUVCxCTK5mL7Q1xE1J2Npcuq8xar8zCPTaP_fHQksJakGq3d-LJw)  
[Understanding Infectious Disease Transmission in Urban Built Environments](https://www.ncbi.nlm.nih.gov/books/NBK507339/?fbclid=IwAR2_IiTsDD-nAapQjVdXc9H0z0e4qWeTEdW-yr_ni-EdkbSjNNX10NDYWQI)
[Mathematical Modeling of Infectious Disease Dynamics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3710332/?fbclid=IwAR36rvyQojKs3p_isAxXUnZDBsmKBt3kH9vaun5H4ap_8G19PsaqaIYULOo)  
[Modeling Infectious Disease Dynamics in the Complex Landscape of Global Health](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4445966/?fbclid=IwAR0O_RMJnFphmZk2gvhLa3eduuT_gzYJFcyxrvP_R07sVfFY3baE-aDut2Q)  

