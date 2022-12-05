standard_residues = [
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V']

standard_residues_three = [
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']

max_asa = {
    'ALA':129.0,'ARG':274.0,'ASN':195.0,'ASP':193.0,'CYS':167.0,
    'GLN':225.0,'GLU':223.0,'GLY':104.0,'HIS':224.0,'ILE':197.0,
    'LEU':201.0,'LYS':236.0,'MET':224.0,'PHE':240.0,'PRO':159.0,
    'SER':155.0,'THR':172.0,'TRP':285.0,'TYR':263.0,'VAL':174.0}

standard_atoms = [
        'N','CA','C','O','OXT','CB',
        'CD','CD1','CD2','CE','CE1',
        'CE2','CE3','CG','CG1','CG2',
        'CH2','CZ','CZ2','CZ3','ND1',
        'ND2','NE','NE1','NE2','NH1',
        'NH2','NZ','OD1','OD2','OE1',
        'OE2','OG','OG1','OH','SG','SD']

residue_1hot = {
        'ALA':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'ARG':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'ASN':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'ASP':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'CYS':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'GLN':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'GLU':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'GLY':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        'HIS':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        'ILE':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
        'LEU':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        'LYS':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        'MET':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        'PHE':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        'PRO':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
        'SER':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
        'THR':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
        'TRP':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        'TYR':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        'VAL':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}

atom_types = {'ALA':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'ARG':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'NE':[0,0,0,1,0,0,0,0,0,0,0],
                     'NH1':[0,0,0,1,0,0,0,0,0,0,0],
                     'NH2':[0,0,0,1,0,0,0,0,0,0,0],
                      'CZ':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CD':[0,0,0,0,0,0,0,0,0,0,1]},
              'ASN':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'ND2':[0,1,0,0,0,0,0,0,0,0,0],
                     'OD1':[0,0,0,0,0,1,0,0,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'ASP':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'OD1':[0,0,0,0,0,0,0,1,0,0,0],
                     'OD2':[0,0,0,0,0,0,0,1,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'CYS':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'SG':[1,0,0,0,0,0,0,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'GLN':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'NE2':[0,1,0,0,0,0,0,0,0,0,0],
                     'OE1':[0,0,0,0,0,1,0,0,0,0,0],
                      'CD':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1]},
              'GLU':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'OE1':[0,0,0,0,0,0,0,1,0,0,0],
                     'OE2':[0,0,0,0,0,0,0,1,0,0,0],
                      'CD':[0,0,0,0,0,0,0,0,1,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1]},
              'GLY':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0]},
              'HIS':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'ND1':[0,0,1,0,0,0,0,0,0,0,0],
                     'NE2':[0,0,1,0,0,0,0,0,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'ILE':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG1':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG2':[0,0,0,0,0,0,0,0,0,0,1],
                     'CD1':[0,0,0,0,0,0,0,0,0,0,1]},
              'LEU':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                     'CD1':[0,0,0,0,0,0,0,0,0,0,1],
                     'CD2':[0,0,0,0,0,0,0,0,0,0,1]},
              'LYS':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'NZ':[0,0,0,0,1,0,0,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CD':[0,0,0,0,0,0,0,0,0,0,1],
                      'CE':[0,0,0,0,0,0,0,0,0,0,1]},
              'MET':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CE':[0,0,0,0,0,0,0,0,0,0,1],
                      'SD':[1,0,0,0,0,0,0,0,0,0,0]},
              'PHE':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE2':[0,0,0,0,0,0,0,0,0,1,0],
                      'CZ':[0,0,0,0,0,0,0,0,0,1,0]},
              'PRO':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,0,1],
                      'CD':[0,0,0,0,0,0,0,0,0,0,1]},
              'SER':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'OG':[0,0,0,0,0,0,1,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'THR':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                     'OG1':[0,0,0,0,0,0,1,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG2':[0,0,0,0,0,0,0,0,0,0,1]},
              'TRP':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE3':[0,0,0,0,0,0,0,0,0,1,0],
                     'CZ2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CZ3':[0,0,0,0,0,0,0,0,0,1,0],
                     'CH2':[0,0,0,0,0,0,0,0,0,1,0],
                     'NE1':[0,0,1,0,0,0,0,0,0,0,0]},
              'TYR':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CG':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CD2':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE1':[0,0,0,0,0,0,0,0,0,1,0],
                     'CE2':[0,0,0,0,0,0,0,0,0,1,0],
                      'CZ':[0,0,0,0,0,0,0,0,0,1,0],
                      'OH':[0,0,0,0,0,0,1,0,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1]},
              'VAL':{  'N':[0,1,0,0,0,0,0,0,0,0,0],
                      'CA':[0,0,0,0,0,0,0,0,1,0,0],
                       'C':[0,0,0,0,0,0,0,0,1,0,0],
                       'O':[0,0,0,0,0,1,0,0,0,0,0],
                     'OXT':[0,0,0,0,0,0,0,1,0,0,0],
                      'CB':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG1':[0,0,0,0,0,0,0,0,0,0,1],
                     'CG2':[0,0,0,0,0,0,0,0,0,0,1]}}

