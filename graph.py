import networkx as nx
import matplotlib.pyplot as plt
import random
import gurobipy as gp
from gurobipy import GRB
import itertools 


BIG_NUMBER = 0xFFFFFFFF

#give a child's edges directions such that only the parent's edge is incoming, and others are outgoing
def give_direction(graph:nx.DiGraph, parent, child, mst_vertices):
    graph = graph.copy()
    to_remove = [(child,parent)] #intialize removal list with the removal of the outgoing edge to the parent
    for e in graph.in_edges(child):
        if e[0] != parent:
            to_remove.append(e)
    for e in graph.out_edges(child):
        if e[1] in mst_vertices:
            to_remove.append(e)
    graph.remove_edges_from(to_remove)
    return graph
def give_direction_inplace(graph:nx.DiGraph, parent, child, mst_vertices):
    to_remove = [(child,parent)] #intialize removal list with the removal of the outgoing edge to the parent
    for e in graph.in_edges(child):
        if e[0] != parent:
            to_remove.append(e)
    for e in graph.out_edges(child):
        if e[1] in mst_vertices:
            to_remove.append(e)
    graph.remove_edges_from(to_remove)
    return graph   

def get_max_depth(tree, root):
    shortest_paths, predecessor_nodes = nx.single_source_dijkstra(tree, root)
    max_depth = 0
    if tree.number_of_nodes() != len(shortest_paths):
        # print("ERROR, no solution exists")
        return BIG_NUMBER
    for p in shortest_paths:
        max_depth = max(max_depth, shortest_paths[p])
    return max_depth

def weigh(tree):
    sum = 0
    for u,v,d in tree.edges(data=True):
        sum += d['weight']
    return sum

def make_dij_tree(graph:nx.graph.Graph, root):
    shortest_paths, predecessor_nodes = nx.single_source_dijkstra(graph, root)
    spt_tree = nx.Graph()
    for node,parent in predecessor_nodes.items():
        if parent is not None and root != node:
            spt_tree.add_edge(parent[-2], node, weight=graph.get_edge_data(parent[-2],node)['weight'])
    if len(shortest_paths) != graph.number_of_nodes():
        spt_tree.add_node(1111) # add a node so that the depth will still be infinite
    return shortest_paths, spt_tree

def mst_remove_path(graph:nx.graph.Graph, root, c, not_from_0=False, remove_edges=False, shrink_path=False, paths_less_than_c=False, choose_big_path=False, max_loops=1000):
    graph = graph.copy()
    mst = nx.minimum_spanning_tree(graph)
    #get the depth of each node
    keep_going = True
    loop_count = 0
    seen_edge_lists = set()
    while keep_going and loop_count < max_loops:
        path_to_break = None
        shortest_paths, predecessor_nodes = nx.single_source_dijkstra(mst, root)
        cur_target = c
        for node in shortest_paths:
            if shortest_paths[node] > cur_target:
                # need to remove the path to this node
                path_to_break = nx.shortest_path(mst, root, node)
                if not choose_big_path:
                    break 
                cur_target = shortest_paths[node]
        if path_to_break is None:
            keep_going = False 
        else:
            candidates = []
            if not_from_0: start = 2
            else: start = 1
            for i in range(start,len(path_to_break)):
                gr = graph.copy()
                gr.remove_edge(path_to_break[i-1], path_to_break[i])
                tree = nx.minimum_spanning_tree(gr)
                if nx.is_connected(tree):
                    if not paths_less_than_c or get_max_depth(tree,root) <= c:
                        if not shrink_path or get_max_depth(tree,root) <= get_max_depth(mst,root):
                            candidates.append((tree,(path_to_break[i-1], path_to_break[i])))
            if len(candidates) == 0:
                print("failed to find tree")
                return mst
            best = BIG_NUMBER
            best_edge = None
            for cand,edge in candidates:
                w = weigh(cand)
                if w < best:
                    mst = cand
                    best = w
                    best_edge = edge
            if loop_count <= 30 or loop_count % 100 == 0:
                print(best, best_edge, loop_count)
            if remove_edges:
                graph.remove_edge(best_edge[0],best_edge[1])
            edges = list(graph.edges()).sort()
            edges = str(edges)
            if edges in seen_edge_lists:
                print("DETECTED REPEAT GRAPH, no solution found")
                return mst
            else:
                seen_edge_lists.add(str(edges))
        loop_count+=1
    if loop_count >= max_loops:
        print("RAN FOREVER")
        return mst
    return mst

def prims_constrained(original_graph:nx.graph.Graph, root, c, predictive=False, relaxation=False, draw_digraph=False):
    graph = original_graph.copy()
    digraph = nx.to_directed(graph)
    mst_vertices = {}
    mst_edges = []
    mst_vertices[root] = 0
    while len(mst_vertices) < graph.number_of_nodes():
        min_edge = None
        min_weight = float('inf')
        min_depth = float('inf')
        for vertex in mst_vertices:
            for _,neighbor,weight in graph.edges(vertex, data='weight'):
                depth = mst_vertices[vertex] + weight
                if neighbor not in mst_vertices and weight < min_weight and depth <= c:
                    if predictive:
                        graph_copy = give_direction(digraph, vertex, neighbor, mst_vertices)
                        new_max_depth = get_max_depth(graph_copy,root)
                    if not predictive or new_max_depth <= c:
                        min_edge = (vertex, neighbor)
                        min_weight = weight
                        min_depth = depth 
        if min_edge is None:
            max_delay_relaxation = 0
            max_edge = None
            if relaxation:
                available_edges = []
                for u in mst_vertices:
                    for v in mst_vertices:
                        if u != v and graph.has_edge(u,v) and v != root:
                            available_edges.append((u,v,graph.get_edge_data(u,v)['weight']))
                for u,v,w in available_edges:
                    for u1,v1 in mst_edges:
                       if v1 == v:
                           ow = graph.get_edge_data(u1,v1)['weight'] 
                           ou = u1
                    depth_relaxation = (ow-w) + mst_vertices[ou]-mst_vertices[u]
                    if depth_relaxation > max_delay_relaxation:
                        max_delay_relaxation = depth_relaxation
                        max_edge = (u,v)
                        max_old_edge = (ou,v)
                        max_w = w
                if max_edge is None:
                    print("failed to find a tree", mst_vertices,mst_edges)
                    return nx.minimum_spanning_tree(graph)
                else:
                    mst_edges.remove(max_old_edge)
                    mst_edges.append(max_edge)
                    uo = max_old_edge[0]
                    u = max_edge[0]
                    v = max_edge[1]
                    mst_vertices[v] -= max_delay_relaxation
                    parents = {v}
                    change = True
                    while change:
                        change = False
                        for u1,v1 in mst_edges:
                            if u1 in parents and v1 not in parents:
                                mst_vertices[v1] -= max_delay_relaxation 
                                parents.add(v1)
                                change = True
            else:
                print("failed to find a tree", mst_vertices, mst_edges)
                pos = nx.spring_layout(digraph)
                plt.subplot(121)
                nx.draw(digraph, pos, with_labels=True, node_size=500, node_color='lightblue', width=2)
                return nx.minimum_spanning_tree(graph)
        else: #min edge exists
            mst_edges.append(min_edge)
            mst_vertices[min_edge[1]] = min_depth
            if predictive: digraph = give_direction(digraph, min_edge[0], min_edge[1], mst_vertices)
    if draw_digraph and predictive:
        pos = nx.spring_layout(digraph,15)
        plt.subplot(121)
        nx.draw(digraph, pos, with_labels=True, node_size=500, node_color='lightblue', width=1)
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(graph.number_of_nodes())])
    for uv in mst_edges:
        graph.add_edge(uv[0], uv[1], weight=original_graph.get_edge_data(uv[0],uv[1])['weight'])
    return graph

def reverse_delete_predictive(original_graph:nx.graph.Graph, root, c):
    graph = original_graph.copy()
    edge_list = list(graph.edges(data=True))
    edge_list.sort(key=lambda x : -x[2]['weight'])
    while len(edge_list) > original_graph.number_of_nodes()-1:
        to_remove = None
        i = 0
        for u,v,d in edge_list:
            graph.remove_edge(u,v)
            if get_max_depth(graph, root) <= c:
                to_remove = (u,v,d)
                break 
            else:
                graph.add_edge(u,v,weight=d['weight'])
            i+=1
        if to_remove is None:
            print("ERROR, nothing left to remove")
            return graph
        edge_list.remove(to_remove)
    return graph
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus

def get_ILP_PuLP(graph:nx.graph.Graph, root, c):
    BIG_CONST = 0
    model = LpProblem("MST_PL", LpMinimize)
    edge_vars = {}
    obj = 0
    for u,v,d in graph.edges(data=True):
        #given that "parents" vars are integers, this is still able to be continuous
        x = LpVariable(name=f"x{(u,v)}", cat="Continuous", lowBound=0)
        edge_vars[(u,v)] = x
        obj += x*d['weight']        
        BIG_CONST += d['weight']
    model += obj

    #a number to subtract such that a sum of a subset of weights will never exceed it
    BIG_CONST*=2 

    #every subset S of nodes all edges add up to at most |S|-1
    nodes = list(graph.nodes())
    for size in range(1,len(nodes)):
        subsets = itertools.combinations(nodes,size)
        for subset in subsets:
            cst = 0
            edge_count = 0
            for u,v in graph.edges():
                if u in subset and v in subset:
                    cst += edge_vars[(u,v)]
                    edge_count += 1
            if edge_count > size-1:
                model += cst <= size-1
    cst = 0
    for edge in graph.edges():
        cst += edge_vars[edge]
    model += cst == len(nodes)-1

    parents = {}
    depths = {}
    for v in nodes:
        depths[v] = LpVariable(name=f"d{(v)}",cat="Continuous",lowBound=0)
        model += depths[v] >= 0
        model += depths[v] <= c

    for v in nodes:
        parent_sum = 0
        for u in nodes:
            if u != v and v != root:
                if graph.has_edge(u,v):
                    x = LpVariable(name=f"{u}->{v}", cat="Integer", lowBound=0)
                    parents[u,v] = x
                    if u != root:
                        if (u,v) in edge_vars: model += edge_vars[(u,v)] >= x
                        else: model += edge_vars[(v,u)] >= x
                    else:
                        if (u,v) in edge_vars: model += edge_vars[(u,v)] == x
                        else: model += edge_vars[(v,u)] == x    
                    parent_sum += x 
        if v != 0: model += parent_sum==1
    for v in nodes:
        for u in nodes:
            if u != v and v != root and u != root:
                if (u,v) in parents:
                    model += parents[u,v] + parents[v,u] <= 1

    #add depth constraints
    for v in nodes:
        for u in nodes:
            if u != v and v != root and graph.has_edge(u,v):
                #if u is vs parent
                w = graph.get_edge_data(u,v)['weight']
                model += depths[v] >= depths[u] + w - (1-parents[u,v])*BIG_CONST
    return model, edge_vars


def get_ILP(graph:nx.graph.Graph, root, c):
    BIG_CONST = 0
    model = gp.Model("MST_PL")
    edge_vars = {}
    obj = 0
    for u,v,d in graph.edges(data=True):
        #given that "parents" vars are integers, this is still able to be continuous
        x = model.addVar(name=f"x{(u,v)}",vtype=GRB.CONTINUOUS)
        edge_vars[(u,v)] = x
        obj += x*d['weight']
        model.addConstr(x >= 0, name=f"c-x{(u,v)}")
        BIG_CONST += d['weight']
    model.setObjective(obj)

    #a number to subtract such that a sum of a subset of weights will never exceed it
    BIG_CONST*=2 

    #every subset S of nodes all edges add up to at most |S|-1
    nodes = list(graph.nodes())
    for size in range(1,len(nodes)):
        subsets = itertools.combinations(nodes,size)
        for subset in subsets:
            cst = 0
            for u,v in graph.edges():
                if u in subset and v in subset:
                    cst += edge_vars[(u,v)]
            model.addConstr(cst <= size-1, name=f"subset-{subset}")
    cst = 0
    for edge in graph.edges():
        cst += edge_vars[edge]
    model.addConstr(cst == len(nodes)-1, name=f"subset-{subset}")

    parents = {}
    depths = {}
    for v in nodes:
        depths[v] = model.addVar(name=f"d{(v)}",vtype=GRB.CONTINUOUS)
        model.addConstr(depths[v] >= 0, name=f"c-d{(v)}g0")
        model.addConstr(depths[v] <= c, name=f"c-d{(v)}")

    for v in nodes:
        parent_sum = 0
        for u in nodes:
            if u != v and v != root:
                if graph.has_edge(u,v):
                    x = model.addVar(name=f"{u}->{v}", vtype=GRB.INTEGER)
                    parents[u,v] = x
                    model.addConstr(x >= 0, name=f"c-x{u}->{v}")
                    if u != root:
                        if (u,v) in edge_vars: model.addConstr(edge_vars[(u,v)] >= x,name=f"c{u}->{v}")
                        else: model.addConstr(edge_vars[(v,u)] >= x,name=f"c{u}->{v}")
                    else:
                        if (u,v) in edge_vars: model.addConstr(edge_vars[(u,v)] == x,name=f"c{u}->{v}")
                        else: model.addConstr(edge_vars[(v,u)] == x,name=f"c{u}->{v}")       
                    parent_sum += x 
        if v != 0: model.addConstr(parent_sum == 1,name=f"1-->{v}")
    for v in nodes:
        for u in nodes:
            if u != v and v != root and u != root:
                if (u,v) in parents:
                    model.addConstr(parents[u,v] + parents[v,u] <= 1,name=f"{u}<-/->{v}")

    #add depth constraints
    for v in nodes:
        for u in nodes:
            if u != v and v != root and graph.has_edge(u,v):
                #if u is vs parent
                w = graph.get_edge_data(u,v)['weight']
                model.addConstr(depths[v] >= depths[u] + w - (1-parents[u,v])*BIG_CONST, name=f"c-{u}->{v}--d{v}")

    return model

def ILP_Solution(graph, root, c):
    m = get_ILP(graph, root, c)
    m.optimize()
    ret = nx.Graph()
    ret.add_nodes_from([i for i in range(graph.number_of_nodes())])
    if m.Status == GRB.OPTIMAL:
        for u,v,d in graph.edges(data=True):
            var = m.getVarByName(f"x{(u,v)}")
            if var is not None and var.X > 10**-5: #sometimes not 0 even though it should be
                ret.add_edge(u,v,weight=d['weight'])
    else:
        for u,v,d in graph.edges(data=True):
            ret.add_edge(u,v,weight=d['weight'])    
    return ret

def ILP_Solution_PuLP(graph, root, c):
    m, edge_vars = get_ILP_PuLP(graph, root, c)
    m.solve()
    ret = nx.Graph()
    ret.add_nodes_from([i for i in range(graph.number_of_nodes())])
    if LpStatus[m.status] == "Optimal":
        for u,v,d in graph.edges(data=True):
            var = edge_vars[(u,v)]
            if var is not None and var.varValue > 10**-5: #sometimes not 0 even though it should be
                ret.add_edge(u,v,weight=d['weight'])
    else:
        for u,v,d in graph.edges(data=True):
            ret.add_edge(u,v,weight=d['weight'])    
    return ret        

def check_validity(tree, c, root):
    depth = get_max_depth(tree,root) 
    if depth > c:
        return False 
    return nx.is_tree(tree)


def get_results(tree_dict, c, root):
    n_str = "  \t "
    w_str = "w \t"
    d_str = "d \t"
    eps_str = "e \t"
    v_str = "v?\t"

    best_valid = None
    best_weight = BIG_NUMBER

    res_dict = {}

    for name in tree_dict:
        n_str += f"{name:8}\t"
        tree = tree_dict[name]
        weight = weigh(tree)
        depth = get_max_depth(tree,root)
        if depth <= c and weight <= best_weight:
            best_weight = weight 
            best_valid = tree
        v = check_validity(tree, c, root)
        if v:
            v_str += "   valid\t"
        else:
            v_str += " invalid\t"
        w_str += f"{weight:8}\t"
        d_str += f"{depth:8}\t"
        res_dict[name] = {'weight':weight, 'depth':depth, 'valid':v}

    for name in tree_dict:
        tree = tree_dict[name]
        weight = weigh(tree)
        eps_str += f"{weight/best_weight:8.7}\t"
        res_dict[name]['epsilon'] = weight/best_weight

    res_dict['best_weight'] = best_weight
    res_dict['c'] = c
    res_dict['root'] = root
    
    return f"{n_str}\n{w_str}\n{d_str}\n{eps_str}\n{v_str}", res_dict, best_valid


def generate_anti_spt_graph(n, eps, k):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(n+1)])
    for i in range(1,n+1):
        G.add_edge(0, i, weight=k)
        G.add_edge(i, i+1, weight=eps)
    return G

def generate_anti_prims_predictive_graph(n, m, k):
    G = nx.Graph()
    G.add_node(0)
    G.add_nodes_from([i for i in range(1,n+1)]) #1-n
    G.add_nodes_from([i for i in range(n+1,m+n+2)])#n+1-->n+m+1
    for i in range(1,n+1):
        G.add_edge(i-1,i, weight=k)
    G.add_edge(0, n, weight=k+1)
    for i in range(n+1,m+n+2):
        G.add_edge(0, i, weight=n*k)
        G.add_edge(i,n, weight=k)
    return G

import copy
import heapq

class A_Star_Candidate:
    def __init__(self, graph:nx.Graph, nodes, edges, root, c) -> None:
        self.G = graph 
        self.edges = edges
        self.nodes = nodes

        self.f = BIG_NUMBER


    def validate(self, root, c):
        # graph = self.G.copy()
        node_depths = {root:0}
        max_depth = 0
        revisit = []
        for u,v,w in self.edges:
            if (v, u, w) in self.edges:
                return False
            revisit.append((u,v,w))
        i = 0
        while len(revisit) > 0:
            (u,v,w) = revisit[i]
            if u in node_depths:
                node_depths[v] = node_depths[u] + w
                max_depth = max(max_depth, node_depths[v])
                revisit.remove((u,v,w))
                i = 0
            elif v in node_depths:
                node_depths[u] = node_depths[v] + w
                max_depth = max(max_depth, node_depths[u])
                revisit.remove((u,v,w))
                i = 0        
            else:
                i+=1
            if i >= len(revisit):
                i = 0
        # d = get_max_depth(graph, root)
        return max_depth <= c

    def score(self, root, c):
        if not self.validate(root, c):
            self.f = BIG_NUMBER
            return
        w = 0
        # for e in self.edges:
        #     w+=e[2]
        # self.f = w
        # since f(n) = g(n) + h(n), and g(n) is the current weight of edges, h(n) is the weight of all other edges in a minimum spanning tree,
        # then f(n) is just the weight of a minimum spanning tree that includes <edges>
        chosen_edges = list(copy.copy(self.edges))
        all_edges = list(self.G.edges(data=True))
        all_edges.sort(key=lambda x : x[2]['weight'])
        included_vertices = set()
        for e in chosen_edges:
            included_vertices.add(e[0])
            included_vertices.add(e[1])
        i = 0
        while len(chosen_edges) < self.G.number_of_nodes()-1 and i < len(all_edges):
            candidate = all_edges[i]
            if candidate[0] not in included_vertices or candidate[1] not in included_vertices:
                chosen_edges.append((candidate[0],candidate[1],candidate[2]['weight']))
            i+=1
        w = 0
        for e in chosen_edges: 
            w += e[2]
        self.f = w
    def expand(self, fringe, root, c, visited):
        for vertex in self.nodes:
            for u,v,d in self.G.edges(vertex, data=True):
                if (v,vertex,d['weight']) not in self.edges:
                    s = {v}.union(self.nodes)
                    l = {(vertex,v,d['weight'])}.union(self.edges)
                    cand = A_Star_Candidate(self.G, s, l,root,c)
                    if cand not in visited:
                        cand.score(root, c)
                        if cand.f < BIG_NUMBER:
                            heapq.heappush(fringe, (cand.f, cand))
                            visited.add(cand)
    
    def __str__(self) -> str:
        edges = []
        for e in self.edges:
            edges.append((max(e[0],e[1]), min(e[0],e[1]),e[2]))
        edges.sort()
        nodes = list(copy.copy(self.nodes))
        nodes.sort()
        return str(nodes) + ":" + str(edges)
    
    def __hash__(self) -> int:
        return hash(str(self))
    def __eq__(self, __value: object) -> bool:
        return str(self) == str(__value)
    def __lt__(self, __value:object) -> bool:
        return self.f < __value.f
    
def a_star_based(graph:nx.Graph, root, c):
    fringe = []
    first_cand = A_Star_Candidate(graph, {root}, set(), root, c)
    first_cand.score(root,c)
    heapq.heappush(fringe, (first_cand.f, first_cand))
    visited = set()
    visited.add(first_cand)

    i = 0
    while len(fringe) > 0:
        pri,cand = heapq.heappop(fringe)
        cand.expand(fringe, root, c, visited)
        if i % 1000 == 0:
            print(cand, "fringe length:", len(fringe))
        i+=1
        if len(cand.nodes) == graph.number_of_nodes():
            break
    g = nx.Graph()
    g.add_nodes_from([i for i in range(graph.number_of_nodes())])
    for u,v,w in cand.edges:
        g.add_edge(u,v,weight=w)
    return g







#used for the improvement phase of BDB
def LBA_algorithm(tree:nx.Graph, graph:nx.Graph, root, c, new_edge_weight, old_edge_weight, child):
    # http://reeves.csc.ncsu.edu/Theses/1996_10_SalamaThesis.pdf page 76
    cycle = nx.cycle_basis(tree, child)
    if len(cycle) != 1:
        print(list(tree.edges()))
        print(cycle, "more or less than one cycle?")
        exit(-1)
    cycle = cycle[0]

    improvement = 0
    l_remove = None
    l_add = None

    for i,xi in enumerate(cycle):
        if i < len(cycle)-1: this_edge = (xi, cycle[i+1])
        else : this_edge = (xi, cycle[0])
        # fine the least cost l'i to replace li in connecting xi upstream towards the source
        min_rep_w = float("inf")
        for u,v,d in graph.edges(xi, data=True):
            if not tree.has_edge(u,v) and d['weight'] < min_rep_w:
                w_l_add = d['weight']
                w_l_remove = tree.get_edge_data(this_edge[0],this_edge[1])['weight']
                if w_l_remove + old_edge_weight - w_l_add - new_edge_weight > improvement:
                    rep_tree = tree.copy()
                    rep_tree.remove_edge(this_edge[0],this_edge[1])
                    rep_tree.add_edge(u, v, weight=w_l_add)
                    if get_max_depth(rep_tree, root) <= c:
                        improvement = +w_l_remove + old_edge_weight - w_l_add - new_edge_weight
                        l_remove = (this_edge[0],this_edge[1])
                        l_add = (u,v)
                        min_rep_w = w_l_add 
    print("LBA reccomends replacing ", l_remove, " with ", l_add, " to handle replacing ", old_edge_weight, " with ", new_edge_weight, " for ", improvement)
    if l_remove is None:
        return False, None, None 
    else:
        return True, l_add, l_remove






def remove_edge_attempt(tree:nx.Graph, graph:nx.Graph, root, c, old_parent, child, new_parent):
    wnew = graph.get_edge_data(child, new_parent)['weight']
    wold = tree.get_edge_data(child, old_parent)['weight']
    if wold <= wnew:
        return False, None, None
    rep_tree = tree.copy()
    rep_tree.remove_edge(old_parent, child)
    rep_tree.add_edge(new_parent, child, weight=wnew)
    has_loop = not nx.is_connected(rep_tree)
    if not has_loop and get_max_depth(rep_tree, root) < c:
        return True, None, None 
    elif has_loop:
        print("HAS LOOP", list(tree.edges()))
        return LBA_algorithm(rep_tree, graph, root, c, wnew, wold, child)
    return False, None, None


#this function should take a tree and a graph and replace edges in the tree with different edges from the graph such that 
# the new tree is still feasible but its weight is lowered as much as possible each step
def improve(tree:nx.Graph, graph:nx.Graph, root, c):
    # http://reeves.csc.ncsu.edu/Theses/1996_10_SalamaThesis.pdf page 76
    keep_going = True 
    while keep_going:
        keep_going = False
        min_weight = float("inf")
        min_edge = None
        min_l_add_remove = None
        depths, predecessor_nodes = nx.single_source_dijkstra(tree, root)
        for u,v,d in graph.edges(data=True):
            if not tree.has_edge(u,v):
                if d['weight'] < min_weight:
                    old_parent_u = None
                    old_parent_v = None
                    v_moved = False 
                    u_moved = False
                    if v != root: old_parent_v = predecessor_nodes[v][-2]
                    if u != root: old_parent_u = predecessor_nodes[u][-2]
                    if old_parent_v is not None: v_moved, l_add, l_remove = remove_edge_attempt(tree, graph, root, c, old_parent_v, v, u)
                    if v_moved:
                        min_l_add_remove = (l_add, l_remove)
                        min_edge = ((u,v),(v,old_parent_v))
                    else:
                        if old_parent_u is not None: u_moved, l_add, l_remove = remove_edge_attempt(tree, graph, root, c, old_parent_u, u, v)
                        if u_moved: 
                            min_l_add_remove = (l_add, l_remove)
                            min_edge = ((u,v),(u,old_parent_u))
        
        if min_edge is not None:
            keep_going = True
            tree = tree.copy()
            min_edge_rep_weight = graph.get_edge_data(min_edge[0][0],min_edge[0][1])['weight']
            tree.remove_edge(min_edge[1][0], min_edge[1][1])
            tree.add_edge(min_edge[0][0],min_edge[0][1],weight=min_edge_rep_weight)
            print("replacing ", min_edge[1], " with ", min_edge[0])
            if min_l_add_remove is not None and min_l_add_remove[0] is not None:
                print("replacing ", min_l_add_remove[1], " with ", min_l_add_remove[0])
                ladd_edge_rep_weight = graph.get_edge_data(min_l_add_remove[0][0],min_l_add_remove[0][1])['weight']
                tree.remove_edge(min_l_add_remove[1][0], min_l_add_remove[1][1])
                tree.add_edge(min_l_add_remove[0][0],min_l_add_remove[0][1],weight=ladd_edge_rep_weight)        

    return tree 
    
def BDB_Heurisitc(graph, root, c):
    # http://reeves.csc.ncsu.edu/Theses/1996_10_SalamaThesis.pdf page 76
    #prims constrained with relaxation instead of predictive (when no edge can be added, instead replace an edge)
    #improvement process -- this process can also be added over the other algorithms
    phase_1 = prims_constrained(graph, root, c, relaxation=True) # already implemented prims, reuse that instead of doing it from scratch
    # phase_2 = improve(phase_1, graph, root, c)
    return phase_1


def RBMH_Heuristic(graph, root, c):
    #https://www.ac.tuwien.ac.at/files/pub/berlakovich-11.pdf
    #gonna be hard to implement, very little information is given
    pass 

def KBH_Heuristic(graph, root, c):
    #https://link.springer.com/chapter/10.1007/978-3-642-04772-5_92
    #https://github.com/Logic4Twisted/RDCMST/blob/master/RDCMST.py
    pass




