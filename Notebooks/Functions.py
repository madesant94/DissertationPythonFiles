#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import networkx as nx
import osmnx as ox

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import matplotlib

from joypy import joyplot

from shapely import geometry

from geopandas.tools import sjoin
from geopandas import GeoDataFrame
import geopandas as gpd

from shapely import geometry
from shapely.geometry import Point
from shapely.geometry import Polygon

import math

from math import radians, cos, sin, asin, sqrt

print('loaded libraries')


#51.493599, -0.185418


# In[ ]:


# Prunning Functions

from collections import defaultdict

def get_degs_freq_dict(G, i):
    """
    just to avoid unweildy code snippet
    G: Graph
    i: iteration (for debugging purpose)
    """
    degs = []
    node_color = []
    edge_color = []
    degs_freq_dict = defaultdict(lambda: [])
    
    for node_id in G.nodes:
        deg = len(G.edges(node_id))
        degs_freq_dict[deg] = np.append(degs_freq_dict[deg], node_id)
        if (deg <= 1):
            node_color.append('gray')
        else:
            node_color.append('gray')
    for e in G.edges:
        n1, n2 = e[0], e[1]
        if (n1 in degs_freq_dict[1]) or (n2 in degs_freq_dict[1]):
            edge_color.append('gray')
        else:
            edge_color.append('gray')
            
    return degs_freq_dict, node_color, edge_color

def get_deletable_nodes(G, i):
    """
    to filter out undeletable nodes 
    (because there are still some edges attached to it)
    G: Graph
    i: iteration (for debugging purpose)
    """
    node_colors = []
    deletable_nodes = []
    degs_freq_dict, _, _ = get_degs_freq_dict(G, i)
    
    # check if there exists some other nodes attach to it
    for node_id in degs_freq_dict[0]:
        deg_cnt = 0
        for e in G.edges:
            if (e[0] != node_id and e[1] == node_id):
                deg_cnt += 1
        # if only 1 node refer to it, we can still delete it
        if (deg_cnt <= 1):
            deletable_nodes.append(node_id)
    
    # check if there exists some other nodes attach to it
    for node_id in degs_freq_dict[1]:
        deg_cnt = 0
        for e in G.edges:
            if (e[0] != node_id and e[1] == node_id):
                deg_cnt += 1
        if (deg_cnt <= 1):
            deletable_nodes.append(node_id)
    
    for node_id in G:
        color = 'gray'
        if node_id in degs_freq_dict[0] or node_id in degs_freq_dict[1]:
            ### uncomment to which nodes we save them from getting deleted
            # color = 'blue'
            color = 'gray'
        if node_id in deletable_nodes:
            color = 'red'
        node_colors.append(color)
            
    return deletable_nodes, node_colors

def prune_culdasacs(G, show_last_result=True, verbose=False):
    i = 0 # iteration
    G_pruned = G.copy() # preserve original graph 
    if verbose:
        print('------- initial state -------')
        print(nx.info(G_pruned))
        print('-----------------------------')
    while(True):
        deletable_nodes, node_colors = get_deletable_nodes(G_pruned, i) 
        if len(deletable_nodes) > 0:
            if verbose:
                ox.plot_graph(G_pruned, node_color=node_colors, node_size=20, fig_height=5, node_alpha=0.8)
            G_pruned.remove_nodes_from(deletable_nodes)
            if verbose:
                print('---- [iter {}] removed {} nodes ----'.format(i, len(deletable_nodes)))
                print(nx.info(G_pruned))
                print('---------------------------------\n')
        else:
            if verbose:
                print('------------ finished -----------')
            if show_last_result:
                ox.plot_graph(G_pruned, node_size=20, fig_height=8, node_alpha=0.8)
            break
        i += 1
    return G_pruned


def get_bbox(point, dist=2000):
    
    earth_radius = 6371000 #meters
    lat, lng = point
    
    delta_lat = (dist / earth_radius) * (180 / math.pi)
    delta_lng = (dist / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180)
    
    north  = lat + delta_lat
    south  = lat - delta_lat
    east = lng + delta_lng
    west = lng - delta_lng
    return north, south, east, west


# get tile Center

def createTileCenters(gridx, gridy):
    """
    inputs gridx, gridy
    return void
    function for creating tiles Center (points of interest)
    """
    tileCenter = {'gpsLatitude': [], 'gpsLongitude':[], 'geometry':[], 'code':[]}
    for i in range(len(gridx) - 1):

        cx = np.linspace(gridx[i], gridx[i+1], 3)

        for j in range(len(gridy)-1):
            cy = np.linspace(gridy[j], gridy[j+1], 3)
            #print('center tile', cy[1],  cx[1])
            tileCenter['gpsLatitude'].append(cy[1])
            tileCenter['gpsLongitude'].append(cx[1])
            tileCenter['geometry'].append(Point(cy[1],cx[1]))
            tileCenter['code'].append((i,j))
        
    tileCenter = pd.DataFrame(tileCenter)
    display(tileCenter.head())
    tileCenter.to_csv('./data/tileCenter.csv')
    print('createTileCentets: Done')
    
def calculateHaversine(lon1, lat1, lon2, lat2):
    #Calculate Haversine Distance
    EARTH_RADIUS = 6371;

    lat1, lon1, lat2, lon2 = map(radians,
                                 [lat1,
                                  lon1,
                                  lat2,
                                  lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    
    return EARTH_RADIUS * c * 1000;

def obtainGridDescription(gridx, gridy):
    """
    return void
    input gridx, gridy
    function to obtain description of the grid (double check)"""
        
    lon1 = gridx[0]
    lon2 = gridx[-1]

    lat1 = gridy[0]
    lat2 = gridy[-1]

    diagonal = calculateHaversine(lon1,lat1, lon2, lat2)

    lenx = calculateHaversine(lon1,lat1, lon2, lat1)
    leny = calculateHaversine(lon1,lat1, lon1, lat2)
    tilex = calculateHaversine(gridx[0],lat1, gridx[1], lat1)
    tiley = calculateHaversine(gridx[0],gridy[0], gridx[0], gridy[1])

    print('Grid X length:', round(lenx,3))
    print('Grid Y length:', round(leny,3))
    print('Grid diagonal length:', round(diagonal,3))

    print('\n   Tile X length', round(tilex,3))
    print('   Tile Y length', round(tiley,3))
    
    
def createPolyDataFrame(grid):
    
    """
    return (GEO) pd.DataDrame polyDataFrame 
    input: dict grid 
    Create DataFrame with Polygons
    """

    polyDataFrame = {'code' : [], 'geometry' : [],'grid' : []}

    for i in grid:

        point1 = (grid[i][0][0], grid[i][1][0])
        point2 = (grid[i][0][0], grid[i][1][1])
        point3 = (grid[i][0][1], grid[i][1][1])
        point4 = (grid[i][0][1], grid[i][1][0])

        poly = geometry.Polygon([point1,point2,point3,point4])

        polyDataFrame['code'].append(i)
        polyDataFrame['geometry'].append(poly)
        polyDataFrame['grid'].append(grid[i])

    polyDataFrame = pd.DataFrame(polyDataFrame)
    # convert to geodataframe
    polyDataFrame = GeoDataFrame(polyDataFrame, crs = "EPSG:4326")
    return polyDataFrame

def createMainGraph(poly, prunning = False, network_type = 'walk'):
    """
    return osmnx multidigraph G
    inputs: polygon poly (squared box from grid)
            prunning (True if we want no culdesacs)
    function to crate Main Graph G
    """

    G_orig = ox.graph_from_polygon(polygon = poly , network_type = network_type)
    print(G_orig)
    
    fig, ax = plt.subplots(figsize=(6,6))
    ox.plot_graph(G_orig, edge_color='#808080', bgcolor='w', edge_alpha=0.2, node_color = "none", ax = ax, show = False)
    plt.title ("Graph Original")
    
    """if prunning == True:
        
            
        G_orig = prune_culdasacs(G_orig, show_last_result=False, verbose=False)
        print(G_orig)
        fig, ax = plt.subplots(figsize=(6,6))
        ox.plot_graph(G_orig, edge_color='#808080', bgcolor='w', edge_alpha=0.2, node_color = "none", ax = ax, show = False)
        plt.title ("Pruned Graph")
        filepath = "./data/squared-grid-prunned.graphml"""
        

    # Add Tolerances
    G_proj = ox.project_graph(G_orig)
    if prunning == True:
        #G = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=5, dead_ends=True)
        G = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=5, dead_ends=True)
    else:
        G = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=5, dead_ends=True)
    
    print(G)
    G_geo = ox.project_graph(G, to_crs='epsg:4326')
    G = ox.project_graph(G, to_crs='epsg:4326')

    
    filepath = "./data/squared-grid.graphml"
    
    fig, ax = plt.subplots(figsize=(6,6))
    ox.plot_graph(G, edge_color='#808080', bgcolor='w', edge_alpha=0.2, node_color = "none", ax = ax, show = False)
    plt.title ("Graph Consolidated")
    
    if prunning == True:
        G = prune_culdasacs(G, show_last_result=False, verbose=False)
        
        fig, ax = plt.subplots(figsize=(6,6))
        ox.plot_graph(G, edge_color='#808080', bgcolor='w', edge_alpha=0.2, node_color = "none", ax = ax, show = False)
        plt.title ("Pruned Graph")
        filepath = "./data/squared-grid-prunned.graphml"
    ox.save_graphml(G, filepath)
    return G, G_proj


def getGraphBasicStats(G_proj):
    """
    return void
    input G_proj
    """
    nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)
    graph_area_m_G = nodes_proj.unary_union.convex_hull.area
    print("what sized area does our network cover in square meters?   ", graph_area_m_G)
    print(ox.basic_stats(G_proj, area=graph_area_m_G, clean_int_tol=15))
    
def checkForMissingNodes(G_orig, G):
    """
    return void
    """
    nodes_orig = ox.graph_to_gdfs(G_orig, edges=False)

    missing_nodes = nodes_orig['x'].isna().sum()
    print('missing nodes total', missing_nodes)

    display(nodes_orig.head())
    G_geo = ox.project_graph(G, to_crs='epsg:4326')
    nodes = ox.graph_to_gdfs(G_geo, edges=False)

    display(nodes.head())
    edges = ox.graph_to_gdfs(G, nodes=False)    
    display(edges.head())
    missing_nodes = nodes['x'].isna().sum()
    print('missing nodes total', missing_nodes)
    print('Must be 0.')
    
    

def createNodesCSV(polyDataFrame, G, route = 'walk', prunning = True):
    """
    return pd.DataFrame nodesInGrid
    inputs: OSMNX multigraph G
    function to create nodesInGrid.csv
    
    """
    crs = "EPSG:4326"
    nodes = ox.graph_to_gdfs(G, edges=False)
    nodes = GeoDataFrame(nodes).set_crs(crs, allow_override=True)
    #check which nodes are in the polygon area
    
    #print("nodes")
    display(nodes.head())
    #print("polyDataFrame")
    display(polyDataFrame.head())
    nodesInGrid = sjoin(nodes, polyDataFrame, predicate='within') 

    print('unique codes', len(nodesInGrid.code.unique()))

    suma = 0

    for i in sorted(nodesInGrid.code.unique()):
        nodesInGrid_ = nodesInGrid[nodesInGrid.code == i ]
        suma += len( nodesInGrid_.index)

    print('Sum should match')
    print('Sum', suma, 'len index',len(nodesInGrid.index))

    nodesInGrid = nodesInGrid[["x","y","geometry","code","grid", "index_right"]] #,"weight" #"code2"
    equivalence = nodesInGrid[["code", "index_right"]]
    #display(nodesInGrid)
    #print("created nodesInGrid.csv")
    
    
    nodesInGrid.to_csv("./data/nodesInGrid.csv")
    
    nodesInGrid_ = nodesInGrid.copy()
    nodesInGrid_ = nodesInGrid_[["x","y","index_right"]] #,"weight" #["x","y","code"]
    nodesInGrid_.reset_index(inplace=True)
    nodesInGrid_.rename(columns={'x': 'latitude', 'y': 'longitude', 'osmid':'id', 'index_right':'code'}, inplace=True) #'code2':'code'
    nodesInGrid_['id'] = nodesInGrid_['id'].astype(str) + '_'
    display(nodesInGrid_)
    title1 = "./data/nodesInGrid.csv"
    title2 = "./data/nodes-in-grid.csv"
    
    if prunning:
        title1 = "./data/nodesInGrid-pruned.csv"
        title2 = "./data/nodes-in-grid-pruned.csv"
    
    nodesInGrid.to_csv(title1, index = False)
    nodesInGrid_.to_csv(title2, index = False)
    return nodesInGrid_

def createCSVFiles(G, network = "walk"):
    
    nodes = ox.graph_to_gdfs(G, edges=False)
    nodes.reset_index(inplace=True)
    nodes['osmid'] = nodes['osmid'].astype("string").astype(str) + '_'
    nodes = nodes[["osmid","y","x", "geometry"]] # dropping elevation too
    nodes.rename(columns={'osmid':'id'}, inplace=True)

    display(nodes.head())

    edges = ox.graph_to_gdfs(G, nodes=False) 
    edges.reset_index(inplace=True)

    #print(edges.columns)
    edges = edges[['u', 'v','length']]
    drop_rows = []

    #print('Original',len(edges.index))
     
    multiedges = []
    n = 0
    for index, row in edges.iterrows():
        # Create graph without loops
        if row['u'] == row['v']:
            drop_rows.append(index)
        # For unidirected graph
        if network == "walk":
            if ([row['v'], row['u']] in multiedges) and (index not in drop_rows):
                drop_rows.append(index)
                n = n+1
        multiedges.append([row['u'], row['v']])
        
    print('total multiedges:', n)

    for i in reversed(drop_rows):
        edges.drop(i,axis= 0,inplace=True)
    #print('Dropped self loops',len(edges.index))

    edges['u'] = edges['u'].astype("string").astype(str) + '_'
    edges['v'] = edges['v'].astype("string").astype(str) + '_'
    #print(edges['turn_degree'].dtype)
    #edges['turn_degree'] = edges['turn_degree'].to_string().replace(',', '_').replace('{', '').replace('}', '')[2:]
    
    edges.rename(columns={'u':'source','v':'target'}, inplace=True)
    display(edges.head())
    return(nodes, edges)

def createGraph(nodes, edges):
    """ with preprocessed edges to avoid self loops*"""
    
    mG = nx.MultiDiGraph()
    
    for index, row in nodes.iterrows():
        mG.add_node(row['id'])
        
    for index, row in edges.iterrows():
        mG.add_edge(row["source"],row["target"])
        
    return(mG)


def createGridxy(north,south, east,west,n_tiles):
    # How many x squares
    gridx = np.linspace(east, west, n_tiles)
    print(gridx)
    # How many y squares
    gridy = np.linspace(south, north, n_tiles)
    print(gridy)

    grid = {}
    i = 0
    j = 0
    for x in range(len(gridx)-1):
        j = 0
        for y in range(len(gridy)-1):
            xx = [gridx[x],gridx[x+1]] #[gridx[0],gridx[-1]]
            yy = [gridy[y],gridy[y]]
            grid[i,j] = xx,[gridy[y],gridy[y+1]]
            j = j +1
        i = i +1
    
    return gridx, gridy, grid

