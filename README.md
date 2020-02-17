### Usage
1. download repo
```
git clone git@github.com:Gangl2016/dev_Tome_OGT.git
```

2. change your directory to repo's

3. Use `python predOGT.py -h` to get following help messages:
```
usage: predOGT.py [-h] [--fasta] [--indir] [-o] [-p]

Predict OGT

optional arguments:
  -h, --help       show this help message and exit
  --fasta          a fasta file containing all protein sequence of a proteome.
  --indir          a directory that contains a list of fasta files. Each fasta
                   file is a proteome. Required for the prediction of OGT for
                   a list of microorganisms. Important: Fasta file names much
                   end with .fasta
  -o , --out       out file name
  -p , --threads   number of threads used for feature extraction, default is
                   1. if set to 0, it will use all cpus available
```

#### Examples
(1) For a given list of organisms:
```
python predOGT.py --indir test/ --out test/predicted_ogt_bayes.csv
```
   
(2) For one organism

```
python predOGT.py --fasta test/caldanaerobacter_subterraneus.fasta
```
