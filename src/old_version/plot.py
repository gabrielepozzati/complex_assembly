import os
import sys
import json
import pickle
import seaborn as sb
import matplotlib.pyplot as plt

path = os.path.abspath(__file__)
src_path = '/'.join(path.split('/')[:-1])
config_path = src_path+'/config.json'
with open(config_path,'r') as config:
    config = json.loads(config.read())

write_path = config['main']['base']+'results/similarities.pkl'
with open(write_path, 'rb') as f:
    pdb_dict = pickle.load(f)

seen_rep = []
homochains = []
for key in pdb_dict:
    if len(key) > 4: continue
    if len(pdb_dict[key]['all_ids']) == 0: continue
    avg_id = sum(pdb_dict[key]['all_ids'])/len(pdb_dict[key]['all_ids'])
    if float(avg_id) == 100.0:
       chain_id = list(pdb_dict[key]['representatives'].keys())[0]
       representative = pdb_dict[key]['representatives'][chain_id]
       if representative not in seen_rep:
           homochains.append(pdb_dict[key]['chain_num'])
           seen_rep.append(representative)

bins = range(1,30)
p = sb.distplot(homochains, bins=bins, kde=False)
plt.xlim([0, 30])
plt.show()

max_ids = []
avg_ids = []
for key in pdb_dict:
    if len(key) > 4: continue
    if len(pdb_dict[key]['all_ids']) == 0: continue
    max_ids.append(pdb_dict[key]['max_id'])
    avg_ids.append(sum(pdb_dict[key]['all_ids'])/len(pdb_dict[key]['all_ids']))

p = sb.jointplot(avg_ids, max_ids)
p.set_axis_labels('Average identity', 'Max. identity', fontsize=16)
plt.show()

ids = []
tmscores = []
for key in pdb_dict:
    if len(key) > 4: continue
    for pair in pdb_dict[key]['TMscores']:
        if pair[3] > 0.5:
            ids.append(pair[2])
            tmscores.append(pair[3])

p = sb.jointplot(ids, tmscores)
p.set_axis_labels('Max. identity', 'TMscore', fontsize=16)
plt.show()


