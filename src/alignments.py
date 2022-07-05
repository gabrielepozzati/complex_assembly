import os
import glob
from Bio.pairwise2 import align
from Bio.Align import substitution_matrices

class Alignments():
    def __init__(self, tmp_folder):
        self.tmp_folder = tmp_folder
        self.blosum62 = substitution_matrices.load("BLOSUM62")


    def pairwise(self, seq1, seq2):
        
        alignment = align.globalds(seq1, seq2, self.blosum62, -11, -1)
        id_acc = [1 for res1, res2 in zip(alignment[0][0], alignment[0][1]) \
                  if res1==res2])
        
        return alignment[0], sum(id_acc)/len(alignment[0][0])


    def format_db(self, binary, input_path, out_path=None):

        if not out_path: out_path = self.tmp_folder+'/seq.db'
        
        if os.path.exists(out_path):
            os.remove(out_path)
            for path in glob.glob(out_path+'*'): os.remove(path)
            print (f'WARNING: {out_path} already exist!'
                    'Cleansed before creating new one!')

        command = [binary, 'createdb', input_path, out_path]
        sp.run(command, stdout=sp.DEVNULL)


    def run_mmseq_map(self, binary, db1, db2, out_path,
                      cov='0.5', seqid='0.0'):

        align_path = f'{self.tmp_folder}/align.db'
        
        command = [
            binary, 'map', db1, db2,
            align_path, self.tmp_folder, '-a', 
            '-c', cov, '--min-seq-id', seqid]
        sp.run(command, stdout=sp.DEVNULL)

        command = [
            binary, 'convertalis', align_path,
            out_path, '--format-output', 'query,target,pident']
        sp.run(command, stdout=sp.DEVNULL)


    def run_foldseek_search(self, binary, db1, db2, out_path):
         
        align_path = f'{self.tmp_folder}/align.db'
        tmalign_path = f'{self.tmp_folder}/tmalign.db'

        command = [
            binary, 'search', db1, db2,
            align_path, self.tmp_folder, '-a']
        sp.run(command, stdout=sp.DEVNULL)

        command = [
            binary, 'aln2tmscore', db1, db2, 
            align_path, tmalign_path]
        sp.run(command, stdout=sp.DEVNULL)

        command = [
            binary, 'createtsv', db1, db2,
            tmalign_path, out_path]
        sp.run(command, stdout=sp.DEVNULL)


