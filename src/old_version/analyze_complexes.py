import os
import glob
import json
import pickle
import igraph as ig
import seaborn as sb
import matplotlib.pyplot as plt
from leafy.graph import Graph
from leafy.search import BFS


def plot_graph(data):
    edges = []
    nodes = list(data.keys())
    GOtoID = {node:idx for idx, node in enumerate(nodes)}
    IDtoGO = {str(idx):node for idx, node in enumerate(nodes)}

    for child in nodes:
        for parent in data[child]['is_a']:
            if parent not in data: continue
            edges.append([GOtoID[parent], GOtoID[child]])

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

def create_graph(data):
    edges = []
    nodes = list(data.keys())
    GOtoID = {node:idx for idx, node in enumerate(nodes)}
    IDtoGO = {str(idx):node for idx, node in enumerate(nodes)}
    
    g = Graph(len(nodes))
    for child in nodes:
        for parent in data[child]['is_a']:
            if parent not in data: continue
            g.add_edge(GOtoID[parent], GOtoID[child])

    return g, GOtoID, IDtoGO


def main():
    #### load config file
    path = os.path.abspath(__file__)
    src_path = '/'.join(path.split('/')[:-1])
    config_path = src_path+'/config.json'
    with open(config_path,'r') as config:
        config = json.loads(config.read())

    tmscore_path = config['main']['base']+'results/tmscores.pkl'
    if os.path.exists(tmscore_path):
        with open(tmscore_path, 'rb') as f:
            tmscores = pickle.load(f)
    
    else:
        prev = -1
        tmscores = {}
        assembly_tmscores = {}
        tm_paths = list(glob.glob(config['main']['base']+'results/tmscores/*'))
        for index, path in enumerate(tm_paths):
    
    ######## calculate progression
            actual = int((float(index+1)/len(tm_paths))*100)
            progression = '# {}% done! '.format(actual)
    
            with open(path) as tm_file:
                for line in tm_file:
                    pdb1 = line.split()[0][:4]
                    chain1 = line.split()[0][-1]
                    pdb2 = line.split()[1][:4]
                    chain2 = line.split()[1][-1]
                    target = pdb2+'_'+chain2
                    tmscore = float(line.split()[2])
    
                    if pdb1 == pdb2 and chain1 == chain2: continue            
                    if pdb1 not in tmscores: tmscores[pdb1] = {}
                    if chain1 not in tmscores[pdb1]: tmscores[pdb1][chain1] = {}
                    tmscores[pdb1][chain1][target] = tmscore
        
        if prev+1 == actual:
            print(progression)
            prev = actual

        write_path = config['main']['base']+'results/tmscores.pkl'
        with open(write_path, 'wb') as f:
            pickle.dump(tmscores, f)

    uni_mapping_path = config['main']['base']+'results/pdb_uni_map.pkl'
    with open(uni_mapping_path, 'rb') as f:
        uniprot_mapping = pickle.load(f)

    go_annotations_path = config['main']['base']+'results/GO_exp_annotations.pkl'
    with open(go_annotations_path, 'rb') as f:
        go_annotations = pickle.load(f)

    go_hierarchy_path = config['main']['base']+'results/GO_hierarchy.pkl'
    with open(go_hierarchy_path, 'rb') as f:
        go_hierarchy = pickle.load(f)

    sub_hierarchy = {}
    for go, item in go_hierarchy.items():
        if item['type'] != 'molecular_function': continue
        sub_hierarchy[go] = item

    graph, GOtoID, IDtoGO = create_graph(sub_hierarchy)
    
    plot_data = {'uni_pairs':[], 'tm-score':[], 'shortest path':[], 'type':[]}
 
    prev = -1
#### cycle query pdb chains
    for index, (pdb_code1, pdb_item) in enumerate(tmscores.items()):
        
        actual = int((float(index+1)/len(tmscores.keys()))*100)
        progression = '# {}% done! '.format(actual)

        if pdb_code1 not in uniprot_mapping: continue
        for chain_code1, chain_item in pdb_item.items():
            if chain_code1 not in uniprot_mapping[pdb_code1]: continue

############ find query chain uniprot code that is annotated in GO
            for code in uniprot_mapping[pdb_code1][chain_code1]:
                if code in go_annotations: 
                    uniprot_code1 = code
                    break
                else: 
                    uniprot_code1 = None

############ find query chain GO annotations
            if uniprot_code1: GOs1 = set(go[1] for go in go_annotations[uniprot_code1])
            else: continue


############ cycle target pdb chains
            for target, tmscore in chain_item.items():
                pdb_code2 = target[0:4]
                chain_code2 = target[-1]
                if pdb_code2 not in uniprot_mapping: continue
                if chain_code2 not in uniprot_mapping[pdb_code2]: continue

################ find target chain uniprot code that is annotated in GO
                for code in uniprot_mapping[pdb_code2][chain_code2]:
                    if code in go_annotations: 
                        uniprot_code2 = code
                        break
                    else: 
                        uniprot_code2 = None

################ find target chain GO annotations
                if uniprot_code2: GOs2 = set(go[1] for go in go_annotations[uniprot_code2])
                else: continue

################ skip identical chain pairs and already analyzed pairs
                if uniprot_code1 == uniprot_code2: continue
                uni_pairF = uniprot_code1+'_'+uniprot_code2
                uni_pairR = uniprot_code2+'_'+uniprot_code1
                if uni_pairF in plot_data['uni_pairs']\
                or uni_pairR in plot_data['uni_pairs']: continue


                sps = []
################ initialize for query chain GOs a BFS search 
                for go1 in GOs1:
                    if go1 not in GOtoID: continue
                    bfs = BFS(graph, GOtoID[go1])
                    bfs.run()

#################### cycle target chain GOs and compute shortest path
                    for go2 in GOs2:
                        if go2 not in GOtoID: continue

                        if go1 == go2: 
                            sps.append(0)
                            continue

                        sp = len(bfs.shortest_path(GOtoID[go2]))
                        sps.append(sp)

                if len(sps) == 0: continue
                plot_data['uni_pairs'].append(uni_pairF)
                plot_data['tm-score'].append(tmscore)
                plot_data['shortest path'].append(min(sps))
                plot_data['type'].append('smallest')

                plot_data['uni_pairs'].append(uni_pairF)
                plot_data['tm-score'].append(tmscore)
                plot_data['shortest path'].append(max(sps))
                plot_data['type'].append('largest')

                plot_data['uni_pairs'].append(uni_pairF)
                plot_data['tm-score'].append(tmscore)
                plot_data['shortest path'].append(sum(sps)/len(sps))
                plot_data['type'].append('average')


        if prev+1 == actual:
            print(progression)
            prev = actual

    write_path = config['main']['base']+'results/shortest_paths.pkl'
    with open(write_path, 'wb') as f:
        pickle.dump(plot_data, f)

    sb.jointplot(x='shortest path', y='tm-score', 
                 hue='type', data=plot_data,
                 xlim=(-0.1,15), ylim=(0,1))
    plt.show()



if __name__ == '__main__':

    main()
            
