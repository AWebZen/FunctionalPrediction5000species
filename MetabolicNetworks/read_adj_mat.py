import numpy as np
import networkx as nx

###################################
#
#Get node names, create mapping
#
###################################

all_nodes = open("all_nodes.txt", "r").read().splitlines()
node_mapping = {i:n for i,n in enumerate(all_nodes)}

###################################
#
#Load adjacency matrix.
#
###################################
"""
The adjacency matrices are made through the union of all nodes with all species, which is the list in all_nodes.txt. Therefore, there are actually some nodes that artificially exist in the network, but are isolated. It may be better to remove them.
"""
m = np.load("adj_mat_species/Enterobacter_hormaechei_subsp._xiangfangensis_eclx.npy", allow_pickle=True).sum() #Random example
G = nx.to_networkx_graph(adj_mat, create_using=nx.DiGraph) #Read adjacency matrix as a directed network
#Relabel nodes to correct ones
G = nx.relabel_nodes(G, node_mapping, copy=False)
#Remove isolated nodes
G.remove_nodes_from(list(nx.isolates(G)))

