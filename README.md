This repository contains the main Python files for the dissertation project Route generation for minimal airborne particulate exposure for walking and cycling trips in London for the Msc in Data Science at the University of Edinburgh 2022-2023 (obtaining a final Distinction mark). 

It includes 2 main folders and the dissertation file:
+ Notebooks: all python files for creating graphs and visualizing some results
+ GCP

## Python folder:
It contains data folder
+ for visualization folder: plots for the report

The .ipynb files are
+ create_square_grid_FINAL: Complete procedure to obtain new graphs from open street maps to .csv files (input for mobile app).
+ Interactive_SquareGrid_withBurroughs, Plot_edges_weights_JavaLogic & SensorsLocation are for visualizing maps.
+ UpdatePollutionValues_Server: You can locally modify the values to the server instead of reabuilding the server, through the app you can then download new data with the "Update Pollution" button.
+ ProcessWeightedDijkstra
+ RecreateRealRoute and ProcessreCreatedRouts... are for recreating the real routes and comparing them with pollution prediction model.
+ Functions.py has all the main functions as the create_square_grid_FINAL.ipynb, used to rebuild graphs in different scripts, but it is easier just to load the presaved graphs with os.load_graphml().

In addition, all libraries installed can be found in the environment.yaml file.

## GCP folder:

Contains the flask application. It receives post (locally change the pollution) or get requests (app) to send the pollution grid. If tried to run locally it will not show any message since it uses a local database from google that loads when it is online (from google.cloud import datastore)

The main steps are (on the command window with cd to the main GCP folder):

+ gcloud init -> log in to your account
+ gcloud app deploy -> deploy application (wil take around 5 minutes)
+ gcloud app browse -> open the location (online) where it was uploaded.

Current link is associated to my google account, it is:

https://cleanestpath-gcr.nw.r.appspot.com/data

There is only the pollution grid data in Json format following this format:
{data:
[tileID1: PM2.5 value1, tileID2: PM2.5 value2, ... , tileID3600: PM2.5 value3600]}

Where tileID1 refers to the pollution grid tile element.

## Dissertation File
Final Msc Dissertation Report (Distinction)
