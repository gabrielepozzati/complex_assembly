import os
import sys

def parse_clustering(clustering_path):

    complexes = {}
    clustering = {}
    with open(clustering_path) as tsv:
        for line in tsv:
            repres = line.split()[0]
            member = line.split()[1].rstrip()
            clustering[member] = repres

            pdb = member.split('_')[0]
            if pdb not in complexes: complexes[pdb] = []
            complexes[pdb].append(member)


    complex_clustering = {}
    for pdb in complexes:
        complex_clustering[pdb] = [clustering[member] for member in complexes[pdb]]
    
    for pdb in complex_clustering: complex_clustering[pdb].sort()

    selected_complexes = []
    selected_clustering = []
    for pdb, clustering in complex_clustering.items():
        if clustering not in selected_clustering:
            selected_clustering.append(clustering)
            selected_complexes.append(pdb)

    representative_chains = []
    for repr_set in selected_clustering:
        representative_chains += repr_set
    representative_chains = set(representative_chains)
    
    print (len(complexes), len(selected_complexes), len(representative_chains))


def main():
    parse_clustering(sys.argv[1])

if __name__ == '__main__':

    main()
