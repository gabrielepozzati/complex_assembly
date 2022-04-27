import os
import json
import errno
import signal
import pickle
import pathlib
import subprocess


def run_foldseek(representative, 
                 query_db, search_db, 
                 ali_path, tmp_path, tsv_path):

    subprocess.run('foldseek createdb {} {}'\
        .format(representative, query_db),
        shell=True)

    subprocess.run('foldseek search {} {} {} {} -a'\
        .format(query_db, search_db, ali_path, tmp_path),
        shell=True)

    try:
        subprocess.run('foldseek aln2tmscore {} {} {} {}'\
            .format(query_db, search_db, ali_path, ali_path+'_tmscore'),
            shell=True, timeout=60)
    except:
        return None

    subprocess.run('foldseek createtsv {} {} {} {}'\
        .format(query_db, search_db, ali_path+'_tmscore', tsv_path),
        shell=True)


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


#### group foldseek inputs from non-redundant seqs (>90% id clustering)
    str_tmp_path = config['main']['tmp']+'structures/'
    if not os.path.exists(str_tmp_path): os.mkdir(str_tmp_path)
    db_folder = config['main']['base']+'/data/foldseek_db/'
    search_db = db_folder+'assemblies'

    prev = -1
    representatives = []
    all_codes = list(pdb_dict.keys())
    for index, (code, sub_dic) in enumerate(pdb_dict.items()):
    
        actual = int((float(index+1)/len(all_codes))*100)
        if actual == prev+1:
            progression = '# {}% done - search db formatting ... '.format(actual)
            prev = actual
    
        if len(code) > 4 or code == 'mono': continue
        for chain, representative in sub_dic['representatives'].items():
            representative = '_'.join(representative.split('_')[:-1])
            if representative == sub_dic['full_path']:
                representatives.append(representative)
                copy_path = str_tmp_path+sub_dic['full_path'].split('/')[-1]
                
                if os.path.exists(copy_path) \
                or os.path.exists(search_db): break
                
                subprocess.run('cp {} {}'\
                    .format(sub_dic['full_path'], str_tmp_path), shell=True)
                break

        print (progression)

#### format foldseek database
    if not os.path.exists(db_folder): os.mkdir(db_folder)
    if not os.path.exists(search_db):
        subprocess.run('foldseek createdb {} {}'\
            .format(str_tmp_path, search_db), shell=True)
    
    query_db = db_folder+'/query'
    alignment_path = config['main']['base']+'/results/align' 
    tmscores_folder = config['main']['base']+'/results/tmscores/'
    if not os.path.exists(tmscores_folder): os.mkdir(tmscores_folder)

    for index, representative in enumerate(representatives):
        code = representative.split('/')[-1].split('.')[0]
        tsv_path = '{}/{}.tsv'.format(tmscores_folder, code)
        if os.path.exists(tsv_path):
            max_index = index

    prev = -1
    for index, representative in enumerate(representatives):
        code = representative.split('/')[-1].split('.')[0]
        tsv_path = '{}/{}.tsv'.format(tmscores_folder, code)

        actual = int((float(index+1)/len(representatives))*100)
        if actual == prev+1:
            progression = '######## {}% done - structural similarity search ... '\
                .format(actual)
            prev = actual
        
        if index <= max_index: continue
        #if os.path.exists(tsv_path): continue

        run_foldseek(representative, 
                     query_db, search_db, 
                     alignment_path, config['main']['tmp'], tsv_path)

        print (progression)

    subprocess.run('rm {}*'.format(alignment_path), shell=True)

if __name__ == '__main__':

    main()
