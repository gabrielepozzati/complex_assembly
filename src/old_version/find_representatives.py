import os
import json
import pickle
import subprocess

def main():
#### load config file
    path = os.path.abspath(__file__)
    src_path = '/'.join(path.split('/')[:-1])
    config_path = src_path+'/config.json'
    with open(config_path,'r') as config:
        config = json.loads(config.read())

#### load similarity file
    data_path = config['main']['base']+config['db_objects']['seq_prefix']+'.pkl'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            pdb_dict = pickle.load(f)
    else: raise RuntimeError ('seqres.pkl file not found in {}!'\
              .format(config['db_objects']['similarities']))

#### write out all sequences in single fasta format file
    fasta = config['main']['base']+config['db_objects']['seq_prefix']+'.fasta'
    with open(fasta, 'w') as out:
        for pdb_code in pdb_dict:
            if len(pdb_code) > 4 or pdb_code == 'mono': continue
            for chain in pdb_dict[pdb_code]['seqres']:
                header = '>{}_{}\n'.format(pdb_dict[pdb_code]['full_path'], chain)
                seq = '{}\n'.format(pdb_dict[pdb_code]['seqres'][chain])
                if seq != '': out.write(header+seq)

#### run mmseq on generated fasta
    tmp = config['clustering']['tmp_path']
    seq_id = config['clustering']['seq_id']
    e_value = config['clustering']['e_value']
    out = config['main']['base']+config['db_objects']['seq_prefix']
    subprocess.run('mmseqs easy-cluster {} {} {} --min-seq-id {} -e {}'\
        .format(fasta, out, tmp, seq_id, e_value), shell=True)

    out_rep = out+'_rep_seq.fasta'
    out_all = out+'_all_seqs.fasta'
    out_clust = out+'_cluster.tsv'
 
    for key in pdb_dict:
        if len(key) > 4 or pdb_code == 'mono': continue
        pdb_dict[key]['representatives'] = {}

#### read clusters and format data/update object
    clusters = {}
    with open(out_clust) as clust_file:
        for line in clust_file:
            representative = line.split()[0]
            member = line.split()[1].rstrip()
            if representative not in clusters: clusters[representative] = []
            clusters[representative].append(line.split()[1])

            pdb = member.split('/')[-1][:-2].split('.')[0]
            path = '/'.join(member.split('/')[:-1])
            chain = member[-1]
            pdb_dict[pdb]['representatives'][chain] = representative

#### save new/updated objects
    with open(out+'_clusters.pkl', 'wb') as f:
            pickle.dump(clusters, f)

    with open(data_path, 'wb') as f:
            pickle.dump(pdb_dict, f)

#### clean up
    subprocess.run('rm '+out_rep, shell=True)
    subprocess.run('rm '+out_all, shell=True)
    subprocess.run('rm '+out_clust, shell=True)

if __name__ == '__main__':

    main()

