#!/usr/bin/env python
# coding: utf-8

# ### Import library

# In[115]:


import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# ### Change working directory

# In[81]:


get_ipython().run_line_magic('cd', '"F:\\Khang\\EVRY\\Transversal\\Training"')

get_ipython().run_line_magic('pwd', '')


# ### Def some useful functions that will be used
# 

# In[3]:


def distance(c1, c2):
  x_dist = (float(c1[8]) - float(c2[8]))**2
  y_dist = (float(c1[9]) - float(c2[9]))**2
  z_dist = (float(c1[10]) - float(c2[10]))**2
  return (x_dist + y_dist + z_dist)**0.5

def flatten(t):
    return [item for sublist in t for item in sublist]


# ## Part 1: Training

# ### 1.1 Read all files in directory and write proccessed files

# In[82]:


for filename in os.listdir():
    with open(filename,'r') as infile:
        l = ''
        for ligne in infile:
            if((ligne[0:6].replace(" ", "") == "ATOM") and (ligne[13:15].replace(" ","") == "C3")):
                   l += ligne
        with open("Proccessed_"+ filename, "w") as outfile:
            outfile.write(l)


# ### 1.2. Calculate distances of structures

# In[83]:


dict_distance = {}
for filename in os.listdir():
    with open(filename,'r') as infile:
        if("Proccessed" in filename):
            atom = []; serial = []; atom_name = []; alt_loc = []; res_name = []; chain_id = []; res_num = []; 
            code_res = []; x = []; y = []; z = []; occ = []; temp_fact = []; ele_symb = []; char_atom = []
            for ligne in infile:
                atom.append(ligne[0:6].replace(" ",""))
                serial.append(ligne[6:11].replace(" ",""))
                atom_name.append(ligne[12:16].replace(" ",""))
                alt_loc.append(ligne[16:17].replace(" ",""))
                res_name.append(ligne[17:20].replace(" ",""))
                chain_id.append(ligne[21:22].replace(" ",""))
                res_num.append(ligne[22:26].replace(" ",""))
                code_res.append(ligne[26:27].replace(" ",""))
                x.append(float(ligne[30:38]))
                y.append(float(ligne[38:46]))
                z.append(float(ligne[46:54]))
                occ.append(float(ligne[54:60]))
                temp_fact.append(float(ligne[60:66]))
                ele_symb.append(ligne[70:78].replace(" ",""))
                char_atom.append(ligne[78:80].replace(" ",""))
            
            df = pd.DataFrame(list(zip(atom, serial, atom_name, alt_loc,
                           res_name, chain_id, res_num,
                           code_res, x, y, z,
                           occ, temp_fact, ele_symb,
                           char_atom)),
               columns =['atom', 'serial', 'atom_name', 'alt_loc',
                           'res_name', 'chain_id', 'res_num',
                           'code_res', 'x', 'y', 'z',
                           'occ', 'temp_fact', 'ele_symb',
                           'char_atom'])

            
            for k in df.chain_id.unique():
              sub_df = df[df.chain_id == k]
              # Only consider intrachain basepairs
              for i in range(1,sub_df.shape[0]):
                for j in range(i):
                     # Only consider residues separated by at least 3 positions
                    if(abs(int(sub_df.iloc[i][6]) - int(sub_df.iloc[j][6])) >= 3):
                      a = str(sub_df.iloc[i][4]).strip() + " - " + str(sub_df.iloc[j][4]).strip()
                      b = a[::-1]
                      x = distance(sub_df.iloc[i], sub_df.iloc[j])
                      if(x <= 20):
                        if(not((a in dict_distance) or (b in dict_distance))):
                            if(a in ['A - A', 'A - U', 'A - C', 'A - G', 'U - U', 'U - C',
                                        'U - G', 'C - C', 'C - G', 'G - G']):
                                dict_distance[a] = [x]
                            elif(b in ['A - A', 'A - U', 'A - C', 'A - G', 'U - U', 'U - C',
                                        'U - G', 'C - C', 'C - G', 'G - G']):
                                dict_distance[b] = [x]
                        elif(a in dict_distance):
                          dict_distance[a].append(x)
                        elif(b in dict_distance):
                          dict_distance[b].append(x)


# In[116]:


dict_distance


# ### 1.3. Calculate the reference frequency (P_ref), observed frequency (P_obs) and the score

# In[84]:


P_ref = {}
P_obs = {}
score = {}
for j in range(20):
    P_ref[str(j) + "-" + str(j + 1)] = len([m for m in flatten(dict_distance.values()) if ((m > j) & (m <= j+1))]) / len(flatten(dict_distance.values()))


for i in dict_distance.keys():
    if(not(i in score.keys())):
        score[i] = []
    for j in range(20):
        s = 0
        a = str(i) + "_" + str(j) + "-" + str(j + 1)
        P_obs[a] = len([m for m in dict_distance[i] if((m > j) & (m <= j+1))])/len(flatten([dict_distance[i]]))
        if((P_ref[str(j) + "-" + str(j + 1)] != 0) & (P_obs[a] != 0)):
            t = -math.log(P_obs[a]/P_ref[str(j) + "-" + str(j + 1)])
            print(i, j, t)
            score[i].append([j,t])


# ### 1.4. Plot the interaction profiles

# In[85]:


from matplotlib.pyplot import figure

figure(figsize=(18, 15), dpi=80)

m = 1
for j in score.keys():
  plt.subplot(4,5,m)
  x = [score[j][i][0] for i in range(len(score[j]))]
  y = [score[j][i][1] for i in range(len(score[j]))]
  plt.title(j)
  plt.axhline(y=0)
  plt.plot(x,y)
  m += 1

plt.show()


# ## Part 2: Compute the total score of each structure

# ### 2.1. Create linear interpolation function for dict from training datasets 
# #### the formula is: yj = (ya - yb)*(j - b)/(a - b) + yb

# In[ ]:


def linear_interpol(key, dict_score, each_dict):
    list_score = []
    for i in each_dict[key]:
        for j in range(len(dict_score[key])):
            if(dict_score[key][j][0] == math.floor(i)-1):
                list_score.append((dict_score[key][j+1][1] - dict_score[key][j][1])*(i - math.floor(i)) +  dict_score[key][j][1])
    
    return list_score


# ### 2.2. Apply the formula to all structures

# In[96]:


score_each_struct = {}
for filename in os.listdir():
    with open(filename,'r') as infile:
        if("Proccessed" in filename):
            each_dict_distance = {}
            atom = []; serial = []; atom_name = []; alt_loc = []; res_name = []; chain_id = []; res_num = []; 
            code_res = []; x = []; y = []; z = []; occ = []; temp_fact = []; ele_symb = []; char_atom = []
            for ligne in infile:
                atom.append(ligne[0:6].replace(" ",""))
                serial.append(ligne[6:11].replace(" ",""))
                atom_name.append(ligne[12:16].replace(" ",""))
                alt_loc.append(ligne[16:17].replace(" ",""))
                res_name.append(ligne[17:20].replace(" ",""))
                chain_id.append(ligne[21:22].replace(" ",""))
                res_num.append(ligne[22:26].replace(" ",""))
                code_res.append(ligne[26:27].replace(" ",""))
                x.append(float(ligne[30:38]))
                y.append(float(ligne[38:46]))
                z.append(float(ligne[46:54]))
                occ.append(float(ligne[54:60]))
                temp_fact.append(float(ligne[60:66]))
                ele_symb.append(ligne[70:78].replace(" ",""))
                char_atom.append(ligne[78:80].replace(" ",""))

            df = pd.DataFrame(list(zip(atom, serial, atom_name, alt_loc,
                           res_name, chain_id, res_num,
                           code_res, x, y, z,
                           occ, temp_fact, ele_symb,
                           char_atom)),
               columns =['atom', 'serial', 'atom_name', 'alt_loc',
                           'res_name', 'chain_id', 'res_num',
                           'code_res', 'x', 'y', 'z',
                           'occ', 'temp_fact', 'ele_symb',
                           'char_atom'])


            for k in df.chain_id.unique():
                sub_df = df[df.chain_id == k]
              #print(sub_df, sub_df.shape[0])
                for i in range(1,sub_df.shape[0]):
                    for j in range(i):
                        if(abs(int(sub_df.iloc[i][6]) - int(sub_df.iloc[j][6])) >= 3):
                          a = str(sub_df.iloc[i][4]).strip() + " - " + str(sub_df.iloc[j][4]).strip()
                          b = a[::-1]
                          x = distance(sub_df.iloc[i], sub_df.iloc[j])
                          if(x <= 20):
                            if(not((a in each_dict_distance) or (b in each_dict_distance))):
                                if(a in ['A - A', 'A - U', 'A - C', 'A - G', 'U - U', 'U - C',
                                            'U - G', 'C - C', 'C - G', 'G - G']):
                                    each_dict_distance[a] = [x]
                                elif(b in ['A - A', 'A - U', 'A - C', 'A - G', 'U - U', 'U - C',
                                            'U - G', 'C - C', 'C - G', 'G - G']):
                                    each_dict_distance[b] = [x]
                            elif(a in each_dict_distance):
                              each_dict_distance[a].append(x)
                            elif(b in dict_distance):
                              each_dict_distance[b].append(x)
            
            # Create a temporary list that save scores of each structure
            m = []
            for u in each_dict_distance.keys():
                t = linear_interpol(u,score, each_dict_distance)
                m.append(sum(t)/len(t))
            
            # Save the sum of scores of each structure to a dict
            score_each_struct[filename[11:-4]] = sum(m)


# In[97]:


score_each_struct


# ### Rank structures by their score

# In[114]:


s = 1
for i in sorted(set(score_each_struct.values()), reverse = True):
    for j in score_each_struct.keys():
        if(score_each_struct[j] == i):
            x = j + " has rank " + str(s) + " with score: " + str(i) 
    s += 1
    print(x)

