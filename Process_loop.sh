#!/bin/bash
START=1
END=290
STEP=10
#SLEEP=600 #Just over 11 Minutes (in seconds)

# Make sure all files to be executed are in on_deck
cd ~/Documents/Python/Tome/dev_Tome_OGT/
for i in $(seq $START $STEP $END) ; do	
    JSTART=$i
    JEND=$[ $JSTART + $STEP -1 ] 
    echo "Submitting from ${JSTART} to ${JEND}"
	# Move 10 files at a time from on_deck to proteomes
	cd ~/Documents/Python/Tome/dev_Tome_OGT/test/on_deck
	mv `ls | head -10` ~/Documents/Python/Tome/dev_Tome_OGT/test/proteomes/
	# execute
	cd ~/Documents/Python/Tome/dev_Tome_OGT/
    python predOGT.py --indir test/proteomes/ --out test/tempResults/predicted_ogt_bayes_${JSTART}.csv
	# Clean up by moving files from proteomes to Processed
	mv -v ~/Documents/Python/Tome/dev_Tome_OGT/test/proteomes/* ~/Documents/Python/Tome/dev_Tome_OGT/test/Processed/
done

# Concatenate all the results
cat ~/Documents/Python/Tome/dev_Tome_OGT/test/tempResults/*.csv >> ~/Documents/Python/Tome/dev_Tome_OGT/test/Results/predicted_ogt_bayes_ALL.csv

# Remove the iterations
rm ~/Documents/Python/Tome/dev_Tome_OGT/test/tempResults/*

# move all the fasta's back to on_deck
mv -v ~/Documents/Python/Tome/dev_Tome_OGT/test/Processed/* ~/Documents/Python/Tome/dev_Tome_OGT/test/on_deck/
