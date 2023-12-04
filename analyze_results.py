import json 


fp = open("results.json", "r")
obj = json.load(fp)
fp.close()

max_primpred = 0
max_spt = 0
max_revdel = 0
min_mst = 10000

for i in obj:
    if 'ilp' in obj[i] and obj[i]['ilp']['epsilon'] != 1:
        print("ILP EPSILON IS NOT 1", obj[i], i)
        exit(-1)
    if 'ilp' in obj[i] and not obj[i]['ilp']['valid']:
        print("ILP IS NOT VALID", obj[i], i)
        exit(-1)
    if 'astar' in obj[i] and obj[i]['astar']['epsilon'] != 1:
        print("ASTAR EPSILON IS NOT 1", obj[i], i)
        exit(-1)
    if 'astar' in obj[i] and not obj[i]['astar']['valid']:
        print("ASTAR IS NOT VALID", obj[i], i)
        exit(-1)
    if not obj[i]['primpred']['valid']:
        print("PRIM PREDICTIVE IS NOT VALID", obj[i], i)
        exit(-1)
    if obj[i]['primpred']['epsilon'] > max_primpred:
        max_primpred = obj[i]['primpred']['epsilon']
    if not obj[i]['spt']['valid']:
        print("SPT IS NOT VALID", obj[i], i)
        exit(-1)
    if obj[i]['spt']['epsilon'] > max_spt:
        max_spt= obj[i]['spt']['epsilon']
    if not obj[i]['revdelpr']['valid']:
        print("REVERSE DELETE PREDICTIVE IS NOT VALID", obj[i], i)
        exit(-1)
    if obj[i]['revdelpr']['epsilon'] > max_revdel:
        max_revdel = obj[i]['revdelpr']['epsilon']
    if obj[i]['mst']['epsilon'] < min_mst:
        min_mst = obj[i]['mst']['epsilon']

print("Greatest error of prim pred is ", max_primpred)
print("Greatest error of spt is ", max_spt)
print("Greatest error of rev del is ", max_revdel)
print("Smallest ratio of mst is ", min_mst)