#!/usr/bin/env python

#  This file is part of Tome
#
#  Tome is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Tome is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Tome.  If not, see <https://www.gnu.org/licenses/>


import sys
import os
import pandas as pd
from Bio import SeqIO
from sklearn.externals import joblib
from collections import Counter
from multiprocessing import Pool, cpu_count
import numpy as np
import subprocess
import argparse



# Esimtation of OGT for organism(s)
################################################################################

def parse_args():
    args = dict()
    for i in range(len(sys.argv)):
        item = sys.argv[i]
        if item.startswith('-'):
            item = item.replace('-','')
            try:args[item] = sys.argv[i+1]
            except:args[item] = ''

            if item == 'h': args['help'] = ''

    for i in range(len(sys.argv)):
        if 'tome' in sys.argv[i]:
            try:
                if sys.argv[i+1] in ['predOGT','getEnzymes']:
                    args['method'] = sys.argv[i+1]
            except: None
            break
    return args

def load_means_stds(predictor):
    means=dict()
    stds=dict()
    features=list()
    for line in open(predictor.replace('pkl','f'),'r'):
        if line.startswith('#'):continue
        cont=line.split()
        means[cont[0]]=float(cont[1])
        stds[cont[0]]=float(cont[2])
        features.append(cont[0])
    return means,stds,features


def load_model():
    path = os.path.dirname(os.path.realpath(__file__)).replace('core','')
    predictor = os.path.join(path,'tome_cool/predictor/model.pkl')
    try:
        model=joblib.load(predictor)
        means,stds,features = load_means_stds(predictor)
    except:
        print('Failed loading the model. Trying to train the model...')
        train_model()
        model=joblib.load(predictor)
        means,stds,features = load_means_stds(predictor)

    return model,means,stds,features

def do_count(seq):
    dimers = Counter()
    for i in range(len(seq)-1): dimers[seq[i:i+2]] += 1.0
    return dimers


def count_dimer(fasta_file,p):
    seqs = [str(rec.seq).upper() for rec in SeqIO.parse(fasta_file,'fasta')]

    if p == 0:num_cpus = cpu_count()
    else: num_cpus = p
    results = Pool(num_cpus).map(do_count, seqs)
    dimers = sum(results, Counter())
    return dict(dimers)

def get_dimer_frequency(fasta_file,p):
    dimers = count_dimer(fasta_file,p)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWXY'
    dimers_fq = dict()

    # this is to remove dimers which contains letters other than these 20 amino_acids,
    # like *
    for a1 in amino_acids:
        for a2 in amino_acids:
            dimers_fq[a1+a2] = dimers.get(a1+a2,0.0)
    number_of_aa_in_fasta = sum(dimers_fq.values())
    for key,value in dimers_fq.items(): dimers_fq[key] = value/number_of_aa_in_fasta
    return dimers_fq

def predict(fasta_file,model,means,stds,features,p):
    dimers_fq = get_dimer_frequency(fasta_file,p)

    Xs = list()
    for fea in features:
        Xs.append((dimers_fq[fea]-means[fea])/stds[fea])

    Xs = np.array(Xs).reshape([1,len(Xs)])

    pred_ogt = model.predict(Xs)[0]
    return np.around(pred_ogt,decimals=2)


def main(args):
    infile = args.fasta
    indir = args.indir

    outf = args.out

    model, means, stds, features = load_model()
    outf.write('FileName\tpredOGT (C)\n')

    if infile is not None:
        pred_ogt = predict(infile,model,means,stds,features,args.threads)
        outf.write('{0}\t{1}\n'.format(infile.split('/')[-1], pred_ogt))

    elif indir is not None:
        for name in os.listdir(indir):
            if name.startswith('.'): continue
            if not name.endswith('.fasta'): continue
            pred_ogt = predict(os.path.join(indir,name),model,means,stds,features,args.threads)
            outf.write('{0}\t{1}\n'.format(name, pred_ogt))
    else: sys.exit('Please provide at least a fasta file or a directory that contains \
    a list of fasta files')
    outf.close()

if __name__ == '__main__':
    parser_ogt = argparse.ArgumentParser(description='Predict OGT')

    parser_ogt.add_argument('--fasta',help='a fasta file containing all protein \
    sequence of a proteome.',metavar='',default=None)

    parser_ogt.add_argument('--indir',help='a directory that contains a list of \
    fasta files. Each fasta file is a proteome. Required for the prediction of OGT\
    for a list of microorganisms. Important: Fasta file names much end with .fasta',
    metavar='',default=None)

    parser_ogt.add_argument('-o','--out',help='out file name',
    type=argparse.FileType('w', encoding='UTF-8'),default=sys.stdout,metavar='')

    parser_ogt.add_argument('-p','--threads',default=1,help='number of threads \
    used for feature extraction, default is 1. if set to 0, it will use all cpus available',
    metavar='')
    
    args = parser_ogt.parse_args()
    print(args)
    
    main(args)
    