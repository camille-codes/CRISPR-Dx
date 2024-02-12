# CRISPR-Dx

The main goal of this project is to develop a Bioinformatics pipeline for designing optimal guide-RNAs for a CRISPR-based diagnostics approach. A bioinformatics pipeline is developed that outputs guide-RNAs for custom Cas enzymes. As a simple and intuitive platform, it can be incorporated into workflows for convenient use of laboratory scientists. 

The pipeline consists of three modules: **Target Identification, Target Evaluation, and Uniqueness Evaluation**. In the Target Identification stage, the program scans the target genome sequences and searches for candidate guide-RNA sites that are common across all targets. In Target Evaluation, the program extracts features from the guide-RNA sequences and runs them through a machine learning algorithm that predicts whether they have high (positive) or low (negative) activity in terms of cleavage efficiency. Lastly, in Uniqueness Evaluation, the program outputs the guide-RNA sequences that is unique to that particular family of virus through comparative analysis. 

## Requirements

**Operating system:** Linux

**Linux Packages:**
* Vienna-RNA
* OffTargets (needs permission from CSIRO to download)

**Python:** Version 3.8 and above

**Python Libraries**
* biopython
* bcbio-gff
* numpy
* pandas
* tensorflow
* keras
* scikit-learn

## Parameters

* *--motif* : Protospacer Adjacent Motif (PAM) or Protospacer Flanking Site (PFS)\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;String characters such as NGG and TTTV representing a short series of nucleotides\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ambiguous nucleotides are accepted.

* *--length* : Length of the target site \
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Integer number representing length of the target site in terms of base pairs (bp)

* *--orientation* : Orientation	\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Integer number representing the location of the PAM or PFS which can be either on the 5’ or 3’ end of the target sequence\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Only numbers 5 or 3 are accepted.

* *--enzyme* : Enzyme	\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Integer number representing the type of Cas enzyme to be used with the guide-RNA\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Only numbers 9, 12, or 13 are accepted.
  
* *--input* : Target sequences\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A Fasta (.fasta) or GZ (.gz) file containing the target sequences.\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The script accepts files with multiple sequences separated by an ID header starting with the symbol “>”. See Figure 15 for a sample of a       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fasta file.
  
* *--offtarget* : Off-target sequences\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A Fasta (.fasta) containing the off-target sequences.\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The script accepts files with multiple sequences separated by an ID header starting with the symbol “>”. See Figure 15 for a sample of a       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fasta file.

* *--output* : Output\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Output file where the detected target sites are printed to be processed in the next module (Target Evaluation).\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Text file format (.txt) is preferred for low memory storage.
  
* *--gff* : Gene annotation file\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Optional: The General Feature Format (.gff) contains the annotation of the target sequences including the start and end location of the\       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;transcriptomes. A compressed version (.gff.tz) 

* *--gene* : Target gene\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Optional: The ID of the target gene region as named in the GFF file\

* *--chrom* : Chromosome\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Optional: The ID of the target chromosome in a genome. 

* *--fm* : Path to FuzzyMatching\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Path to where the FuzzyMatching package has been compiled in the computer\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**NOTE:** FuzzyMatching is a Linux package exclusive to **CSIRO**. Only authorised people can get access to this package. 

## Running the program

In this first example given below, two target genomes are provided in the form of two variants of the SARS-COV-2 virus with file names “SARSCOV2a.fasta” and “SARSCOV2b.fasta.” The PAM consists of the nucleotide sequence TTTV (V being either a A, G, or C) located on the 5’ end of the target sequence. The length of the targets for guide-RNA binding site must be 20 bp long. The path to call the FuzzyMatching package is “/home/user1/offtargets” to initiate off-target sequence comparison.

> python masterScript.py --input SARSCOV2a.fasta SARSCOV2b.fasta --motif TTTV --enzyme 12 --length 20 --orientation 5 --output common_strands.txt --offtarget Influenza.fasta --fm /home/user1/offtargets

To target transcriptome sequences i.e., coding regions of a genome, optional input
arguments may be provided. The user has to pass the filename of the gene annotation file (.gff) which contains information on location numbers of the transcriptome regions in the target genomes. To target a specific gene, the gene ID from the .gff file also has to be passed as an argument. Below is an example usage for using these optional arguments, where the gene annotation file is named “GCF_009858895.2_ASM985889v3_genomic.gff.gz” and the gene target is specified as “gene-GU280_gp08.” 

> python masterScript.py --input SARSCOV2a.fasta SARSCOV2b.fasta --motif TTTV --enzyme 12 --length 20 --orientation 5 --output common_strands.txt
--gff GCF_009858895.2_ASM985889v3_genomic.gff.gz --gene gene-GU280_gp08 --fm /home/user1/offtargets

An argument for targeting a specific chromosome of a genome was also created (--chromosome). This is useful for detecting multi-chromosomal bacteria in the case of diagnosing infectious diseases caused by bacterial organisms. The input file is expected to be a Fasta file containing the sequences of a target specie including the core chromosome/s and other extrachromosomal genomes i.e., plasmids, that may exist in a bacterium. 


