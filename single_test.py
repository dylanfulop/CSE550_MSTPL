import networkx as nx
import matplotlib.pyplot as plt
import random
import gurobipy as gp
from gurobipy import GRB
import itertools 
from graph import *
from many_tests import *

DRAW_DIGRAPH = False
DRAW = True
    
max_diff = 0

seed = random.randint(0,10000)
seed = 142
all_trees, G, c, root = perform_trial(seed, min_n=5, max_n=100)



print(f"c = {c} | root = {root}")
res_str, res_dict, best_valid = get_results(all_trees, c, root)
print(res_str)

# ilp = ILP_Solution(G, root, c)
# m = get_ILP(G, root, c)
# m.optimize()
# if m.Status == GRB.OPTIMAL:
#     for u,v,d in G.edges(data=True):
#         var = m.getVarByName(f"x{(u,v)}")
#         if var is not None and var.X != 0 and var.X != 1:
#              print("NOT 1 OR 0:", f"x{(u,v)}=", var.X)


if DRAW:
    pos = nx.spring_layout(G)
    if DRAW_DIGRAPH: plt.subplot(122)
    print(best_valid.edges())
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', width=2)
    nx.draw_networkx_edges(G, pos, edgelist=all_trees['BDB'].edges(), edge_color='red', width=2)
    nx.draw_networkx_edges(G, pos, edgelist=all_trees['BDB_p1'].edges(), edge_color='blue', width=1)
    edge_weights = {(u,v):d['weight'] for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    plt.show()
