import subprocess
import re


def seq2fasta(df_common, dataset):
    '''
    convert seq to fasta format
    :param files:
    :param dataset:
    :return:
    '''

    for file, data in zip(df_common, dataset):
        with open(data+"_MFE.txt", "a") as f:
            for i, seq in enumerate(file['Sequence']):
                print(">seq_"+data+"_"+str(i),file=f)
                print(seq, file=f)

    return (dataset[0]+'_MFE.txt')

def seq2fasta_common(df_common, dataset):
    '''
    convert seq to fasta format
    :param files:
    :param dataset:
    :return:
    '''

    #for file, data in zip(df_common, dataset):
    with open("fasta_high_RNA.txt", "a") as f:
        for i, seq in enumerate(df_common['guide_seq']):
            print(">seq_high_"+str(i),file=f)
            print(seq, file=f)

    #return (dataset[0]+'_MFE.txt')

def mfe2df(file):
    '''
    extract MFE data from RNAfold files to list
    :return:
    '''

    textfile = open(file, 'r')
    filetext = textfile.read()
    textfile.close()

    matches = re.findall('\[(.*?)\]', filetext)

    return matches


def target_MFE(input_file, output_file):
    """
    calls RNAfold and calculates MFE
    :param input_file:
    :param output_file:
    :return:
    """

    subprocess.run("RNAfold -p -d2 --noLP < " + input_file + " > " + output_file, shell=True)

