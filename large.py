import numpy as np
import heapq
import sets
import copy
import random
import math

def build_tables(filename):
    f = open(filename, 'r')
    sa_to_sp = {}
    sa_to_r = {}
    s_to_a = {}
    sp_prev = -1
    f.readline()
    states = set([])
    transitions = []
    for line in f:
        sarsp = line.split(",")
        s = int(sarsp[0])
        a = int(sarsp[1])
        sa = (s, a)
        r = int(sarsp[2])
        sp = int(sarsp[3])
        states.add(s)
        states.add(sp)
        transitions.append([s, a, r, sp])

        if sa in sa_to_sp: sa_to_sp[sa].add(sp)
        else: sa_to_sp[sa] = set([sp]) 

        sa_to_r[sa] = r #check is rewards are the same!
        
        if s in s_to_a: s_to_a[s].add(a)
        else: s_to_a[s] = set([a]) 
        
        if (s != sp_prev) and (sp_prev) != -1: 
            states.add(sp_prev)
        sp_prev = sp 
    return sa_to_sp, sa_to_r, s_to_a, states, transitions

def find_nearest_state(s, seen_states):
    if s in seen_states: return s
    distance = 50000001
    closest = -1
    for seen in seen_states:
        if seen > closest + distance: return closest
        if abs(s - seen) <= distance:
            distance = abs(s - seen)
            closest = seen
    return closest

def norm(v1, v2):
    #if not states: return 0
    norm = 0
    for index in v1: norm = norm + (v1[index] - v2[index])**2
    return np.sqrt(norm)



def sarsa_lam(sa_to_sp, sa_to_r, s_to_a, seen_states, transitions, num_states, alpha=0.1, gamma=0.95, lam=0.8):
    u = dict((states, 0) for states in range(num_states))
    pi = dict((states, 0) for states in range(num_states))
    q = {}
    q_old = {}
    for s in range(num_states):
        for a in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            q[(s, a)] = 0
            q_old[(s, a)] = 0
    zeros = copy.deepcopy(q)
    N = {}#copy.deepcopy(q)
    analyzed_states = set()
    k = len(transitions)
    print("states initialized! start learning...")

    # Sarsa(lambda)
    for t in range(len(transitions)):
        # At the kth iteration, check for convergence. In this case, we go through all samples
        if (t%k == 0): 
            if t != 0: print(t)
            q_old_norm = norm(q_old, zeros)
            E = norm(q, q_old)/max(q_old_norm, 1E-06)
            if E < 0.01 and q_old_norm != 0: break
            q_old = copy.deepcopy(q)
            
        st = transitions[t][0]
        at = transitions[t][1]
        rt = transitions[t][2]
        snext = transitions[t][3]
        analyzed_states.add(st)
        qnext = 0
        if t+1 <= len(transitions) -1:
            if snext == transitions[t+1][0]: 
                anext = transitions [t+1][1]
                qnext = q[(snext, anext)] 
        
        delta = rt + gamma*qnext - q[(st, at)]
        N[(st, at)] = N.get((st, at), 0) + 1
        for s in analyzed_states:
            for a in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                q[(s, a)] = q[(s, a)] + alpha*delta*N.get((s, a), 0) 
                N[(s, a)] = gamma*lam*N.get((s, a), 0)
            #find optimal policy
            #pi[s] = 1+np.argmax([q[(s, a)] for a in [1, 2, 3, 4, 5, 6, 7, 8, 9]])
            if t+1 <= len(transitions) -1:
                if snext != transitions[t+1][0]: 
                    analyzed_states = set() 
                    N.clear()
    
    print("learning done!")
    print("computing policy...")
    
    for s in range(num_states):
        #if s in seen_states: continue 
        NN = find_nearest_state(s, seen_states)
        #for a in [1, 2, 3, 4, 5, 6, 7, 8, 9]: q[(s, a)] = q[(NN, a)]
        pi[s] = 1+np.argmax([q[(NN, a)] for a in [1, 2, 3, 4, 5, 6, 7, 8, 9]])       
        
    return pi

sa_to_sp, sa_to_r, s_to_a, seen_states, transitions = build_tables("large.csv")
print("data extracted!")
num_states = 312020
pi = sarsa_lam(sa_to_sp, sa_to_r, s_to_a, seen_states, transitions, num_states)
print("done! writing...")
f = open("large.policy", 'w')
for s in range(num_states):
    f.write(str(pi[s]))
    f.write('\n')
f.close()

print("policy written :)")






