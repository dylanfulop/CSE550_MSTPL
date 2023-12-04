from graph import *
import json


GRAPH_TYPE_GNM = 1
GRAPH_TYPE_GNP = 2
GRAPH_TYPE_STROGRATZ = 3
GRAPH_TYPE_BARABASI = 4
GRAPH_TYPE_REGULAR = 5
GRAPH_TYPE_PARTITION = 6
GRAPH_TYPE_GEO = 7
GRAPH_TYPE_ANTI_SPT = 8
GRAPH_TYPE_ANTI_PP = 9

def get_random_graph(min_n=5, max_n=10, seed=0, include_anti_graphs=True):

    if include_anti_graphs: type_of_g = random.randint(1,9)
    else: type_of_g = random.randint(1,7)
    if type_of_g == GRAPH_TYPE_GNM:
        n = random.randint(min_n, max_n)
        m = random.randint(n, n**2)
        G = nx.gnm_random_graph(n,m,seed=seed)
        for u,v,d in G.edges(data=True):
            d['weight'] = random.randint(1,1000)
        return G
    elif type_of_g == GRAPH_TYPE_GNP:
        n = random.randint(min_n, max_n)
        p = random.uniform(0.1, 1)
        G = nx.gnp_random_graph(n, p, seed=seed)
        for u,v,d in G.edges(data=True):
            d['weight'] = random.randint(1,1000)
        return G
    elif type_of_g == GRAPH_TYPE_STROGRATZ:
        n = random.randint(min_n, max_n)
        p = random.uniform(0.1, 1)
        k = random.randint(1,n)
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        for u,v,d in G.edges(data=True):
            d['weight'] = random.randint(1,1000)
        return G
    elif type_of_g == GRAPH_TYPE_BARABASI:
        n = random.randint(min_n, max_n)
        m = random.randint(1, n-1)
        G = nx.barabasi_albert_graph(n,m,seed=seed)
        for u,v,d in G.edges(data=True):
            d['weight'] = random.randint(1,1000)
        return G
    elif type_of_g == GRAPH_TYPE_REGULAR:
        n = random.randint(min_n, max_n)
        d = random.randint(1, n-1)
        if n*d % 2 != 0:
            d+=1
        G = nx.random_regular_graph(d,n,seed=seed)
        for u,v,d in G.edges(data=True):
            d['weight'] = random.randint(1,1000)
        return G  
    elif type_of_g == GRAPH_TYPE_PARTITION:
        n = random.randint(min_n, max_n)
        sizes = []
        left = n
        while left > 0:
            next_size = random.randint(1,left)
            sizes.append(next_size)
            left -= next_size
        p_in = random.uniform(0.1, 1)
        p_out = random.uniform(0.1, 1)
        G = nx.random_partition_graph(sizes,p_in,p_out,seed=seed)
        for u,v,d in G.edges(data=True):
            d['weight'] = random.randint(1,1000)
        return G  
    elif type_of_g == GRAPH_TYPE_GEO:
        n = random.randint(min_n, max_n)
        r = random.uniform(1,3)
        G = nx.random_geometric_graph(n, r)
        for u,v,d in G.edges(data=True):
            d['weight'] = random.randint(1,1000)
        return G
    elif type_of_g == GRAPH_TYPE_ANTI_SPT:
        n = random.randint(min_n, max_n)
        k = random.randint(2, 10000)
        eps = random.randint(1,k-1)
        print("SPT")
        G = generate_anti_spt_graph(n-2, eps, k)
        return G
    elif type_of_g == GRAPH_TYPE_ANTI_PP:
        n = random.randint(min_n, max_n-2)
        m = random.randint(1,max_n-n-1)
        k = random.randint(1, 1000)
        if m + n > max_n - 2:
            if m < n: n-=1
            else: m-=1
        print("PP")
        G = generate_anti_prims_predictive_graph(n,m,k)
        return G
    else:
        print("HUH????")
def get_a_graph(min_n,max_n,seed):
    keep_going = True
    while keep_going:
        G = get_random_graph(min_n=min_n,max_n=max_n,seed=seed)
        if G.number_of_nodes() >= 1:
            root = G.nodes()[0]
            if type(root) is dict:
                print(root, G, list(G.nodes()))
                root = 0
            d = get_max_depth(G, root)
            if d < BIG_NUMBER:
                keep_going = False
    return G, root

def perform_trial(seed, min_n=3, max_n=10):
    random.seed(seed)
    print(f"Seed = {seed}")
    G, root = get_a_graph(min_n,max_n,seed)
    spaths, spt = make_dij_tree(G, root)
    max_d = get_max_depth(spt,root)
    mst = nx.minimum_spanning_tree(G)
    mst_d = get_max_depth(mst,root)
 
    c = random.randint(max_d, mst_d)

    all_trees = {}
    all_trees['spt'] = spt 
    all_trees['mst'] = mst

    #arbitrary path choice
    all_trees['mst_rp'] = mst_remove_path(G, root, c, not_from_0=True,remove_edges=False,shrink_path=False,paths_less_than_c=False,choose_big_path=False)
    all_trees['mst_rpe'] = mst_remove_path(G, root, c, not_from_0=True,remove_edges=True,shrink_path=False,paths_less_than_c=False,choose_big_path=False)
    all_trees['mst_rpsp'] = mst_remove_path(G, root, c,not_from_0=True,remove_edges=False,shrink_path=True,paths_less_than_c=False,choose_big_path=False)
    all_trees['mst_rpc'] = mst_remove_path(G, root, c,not_from_0=True,remove_edges=False,shrink_path=False,paths_less_than_c=True,choose_big_path=False)
    #greedy path choice
    all_trees['mst_rpg'] = mst_remove_path(G, root, c,not_from_0=True,remove_edges=False,shrink_path=False,paths_less_than_c=False,choose_big_path=True)
    all_trees['mst_rpeg'] = mst_remove_path(G, root, c,not_from_0=True,remove_edges=True,shrink_path=False,paths_less_than_c=False,choose_big_path=True)
    all_trees['mst_rpspg'] = mst_remove_path(G, root, c,not_from_0=True,remove_edges=False,shrink_path=True,paths_less_than_c=False,choose_big_path=True)
    all_trees['mst_rpcg'] = mst_remove_path(G, root, c,not_from_0=True,remove_edges=False,shrink_path=False,paths_less_than_c=True,choose_big_path=True)

    #not mst_rp based
    all_trees['primc'] = prims_constrained(G, root, c)
    all_trees['primpred'] = prims_constrained(G, root, c, True, draw_digraph=False)
    all_trees['revdelpr'] = reverse_delete_predictive(G, root, c)

    #ILP based
    if G.number_of_nodes() <= 10: all_trees['ilp'] = ILP_Solution(G, root, c) #disallowed without license at 11
    elif G.number_of_nodes() <= 15: all_trees['ilp'] = ILP_Solution_PuLP(G, root, c) #unreasonably slow at 16
    #astar based
    # if G.number_of_nodes() <= 11: #unreasonably slow at 12
    #  all_trees['astar'] = a_star_based(G, root, c)

    return all_trees, G, c, root

def main():
    results = {}
    for i in range(1000):
        all_trees, G, c, root = perform_trial(seed=i, min_n=5, max_n=100)
        print(f"c = {c} | root = {root}")
        res_str, res_dict, best_valid = get_results(all_trees, c, root)
        print(res_str)
        print(f"Seed = {i}")
        results[i] = res_dict

        fp = open("results.json", "w")
        json.dump(results, fp)
        fp.close()
    





if __name__ == "__main__":
    main()

