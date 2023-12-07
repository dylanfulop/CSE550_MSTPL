import json 


fp = open("results.json", "r")
obj = json.load(fp)
fp.close()

max_primpred = 0
max_spt = 0
max_revdel = 0
min_mst = 10000


spt = []
mp = []
mrd = []
bdb_p1 = []
spt_imp = []
mp_imp = []
mrd_imp = []
bdb = []


for i in obj:
    best_score = 1000
    best = "none"
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
    if 'BDB' in obj[i] and not obj[i]['BDB']['valid']:
        print("BDB IS NOT VALID", obj[i], i)
        exit(-1)
    if 'BDB_p1' in obj[i] and not obj[i]['BDB_p1']['valid']:
        print("BDB phase 1 IS NOT VALID", obj[i], i)
        exit(-1)
    if 'BDB' in obj[i] and 'BDB_p1' in obj[i]:
        improvement = obj[i]['BDB_p1']['weight'] - obj[i]['BDB']['weight']
        bdb.append(obj[i]['BDB']['epsilon'])
        bdb_p1.append(obj[i]['BDB_p1']['epsilon'])
        if improvement < 0:
            print("ERROR, IMPROVE MADE WORSE")
            print(obj[i], i)
            exit(-1)
    else:
        print("ERROR unexpected missing stuff")
        exit(-1)
    if 'primpred' in obj[i] and 'pp_imp' in obj[i]:
        improvement = obj[i]['primpred']['weight'] - obj[i]['pp_imp']['weight']
        mp.append(obj[i]['primpred']['epsilon'])
        mp_imp.append(obj[i]['pp_imp']['epsilon'])
        if improvement < 0:
            print("ERROR, IMPROVE MADE WORSE")
            print(obj[i], i)
            exit(-1)
    else:
        print("ERROR unexpected missing stuff")
        exit(-1)
    if 'revdelpr' in obj[i] and 'rd_imp' in obj[i]:
        improvement = obj[i]['revdelpr']['weight'] - obj[i]['rd_imp']['weight']
        mrd.append(obj[i]['revdelpr']['epsilon'])
        mrd_imp.append(obj[i]['rd_imp']['epsilon'])
        if improvement < 0:
            print("ERROR, IMPROVE MADE WORSE")
            print(obj[i], i)
            exit(-1)
    else:
        print("ERROR unexpected missing stuff")
        exit(-1)
    if 'spt' in obj[i] and 'spt_imp' in obj[i]:
        improvement = obj[i]['spt']['weight'] - obj[i]['spt_imp']['weight']
        spt.append(obj[i]['spt']['epsilon'])
        spt_imp.append(obj[i]['spt_imp']['epsilon'])
        if improvement < 0:
            print("ERROR, IMPROVE MADE WORSE")
            print(obj[i], i)
            exit(-1)
    else:
        print("ERROR unexpected missing stuff")
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



import numpy as np
def stats(name, array):
    avg = np.mean(array)
    max = np.max(array)
    med = np.median(array)
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    c1 = 0
    c11 = 0
    for a in array:
        if a == 1:
            c1+= 1
        if a <= 1.1:
            c11+=1
    
    print(f"{name}, {avg:6.5}, {max:6.5}, {med:6.5}, {q1:6.5}, {q3:6.5}, {c1}, {c11}")

stats("spt", spt )
stats("mp", mp)
stats("mrd", mrd)
stats("bdbp1", bdb_p1)

stats("spt_imp", spt_imp)
stats("mp_imp", mp_imp)
stats("mrd_imp", mrd_imp)
stats("bdb", bdb)

sptc = 0
mpc = 0
mrdc = 0
bdbp1c = 0

sptimpc = 0
mpimpc = 0
mrdimpc = 0
bdbc = 0

for i in range(len(bdb)):
    m = min([spt_imp[i], mp_imp[i], mrd_imp[i], bdb[i]])
    if spt_imp[i] == m:
        sptimpc+=1
    if spt[i] == m:
        sptc+=1
    if mrd_imp[i] == m:
        mrdimpc+=1
    if mrd[i] == m:
        mrdc+=1

    if mp[i] == m:
        mpc+=1
    if mp_imp[i] == m:
        mpimpc+=1
    if bdb[i] == m:
        bdbc+=1
    if bdb_p1[i] == m:
        bdbp1c+=1

print("times best")
print(f"{sptc}, {mpc}, {mrdc}, {bdbp1c}, {sptimpc}, {mpimpc}, {mrdimpc}, {bdbc}")

# print("Greatest error of prim pred is ", max_primpred)
# print("Greatest error of spt is ", max_spt)
# print("Greatest error of rev del is ", max_revdel)
# print("Smallest ratio of mst is ", min_mst)



