import os
import json
import errno
import signal
import pickle
import pathlib
import subprocess
import igraph as ig

allowed_codes = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP',
                 'HTP', 'HDA', 'HMP', 'HGI', 'HEP',
                 'TAS', 'RCA', 'IC']


def plot_graph(data):
    edges = []
    nodes = list(data.keys())
    GOtoID = {node:idx for idx, node in enumerate(nodes)}
    IDtoGO = {str(idx):node for idx, node in enumerate(nodes)}
    
    for child in nodes:
        for parent in data[child]['is_a']:
            if parent not in data: continue
            edges.append([GOtoID[parent], GOtoID[child]])

#### init graph
    g = ig.Graph(n=len(nodes), edges=edges, directed=True)
    g.vs["name"] = nodes

    visual_style = {}
    visual_style['vertex_size'] = 500
    visual_style['edge_width'] = 1
    visual_style['edge_arrow_size'] = 1
    visual_style['edge_arrow_width'] = 1

    visual_style['layout'] = g.layout('rt_circular')
    visual_style['bbox'] = (100000, 100000)


    ig.plot(g, **visual_style,
            target='/home/gabriele/Desktop/graph.pdf')


def main():

#### load config file
    path = os.path.abspath(__file__)
    src_path = '/'.join(path.split('/')[:-1])
    config_path = src_path+'/config.json'
    with open(config_path,'r') as config:
        config = json.loads(config.read())

#### parse GO hierarchy file
    hierarchy = {}
    ontology = config['main']['base']+'data/go.obo'
    with open(ontology) as gofile:
        all_relations = ''.join([line for line in gofile])
    
    all_relations = all_relations.split('[Term]\n')[1:]
    all_relations[-1] = all_relations[-1].split('[Typedef]\n')[0]
    for entry in all_relations:

        entry = entry.split('\n')
        for line in entry:
            if line.startswith('id:'): 
                code = line.split()[1].strip()
                hierarchy[code] = {'is_a':[], 'part_of':[]}

            elif line.startswith('namespace:'): 
                gotype = line.split()[1].strip()
                hierarchy[code]['type'] = gotype

            elif line.startswith('is_a:'):
                parent = line.split()[1].strip()
                hierarchy[code]['is_a'].append(parent)

            elif line.startswith('relationship: part_of'):
                parent = line.split()[2].strip()
                hierarchy[code]['part_of'].append(parent)
            
            elif line == 'is_obsolete: true':
                del(hierarchy[code])
                break

    outpath = config['main']['base']+'results/GO_hierarchy.pkl'
    with open(outpath,'wb') as f:
        pickle.dump(hierarchy, f)

#### parse GO sequence annotations
    exp_annotations = {}
    annotations = config['main']['base']+'data/goa_uniprot_all_noiea.gaf' 
    with open(annotations) as gofile:
        for line in gofile: 
            db = line.split()[0]
            uni_id = line.split()[1]
            relation = line.split()[3]
            go_term = line.split()[4]
            evidence = line.split()[6]
            if evidence in allowed_codes and db == 'UniProtKB': 
                if uni_id not in exp_annotations:
                    exp_annotations[uni_id] = []
                exp_annotations[uni_id].append([relation, go_term])

    outpath = config['main']['base']+'results/GO_exp_annotations.pkl'
    with open(outpath,'wb') as f:
        pickle.dump(exp_annotations, f)


    
if __name__ == '__main__':

    main()
                
