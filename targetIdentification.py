from BCBio import GFF
from Bio import SeqIO
import re
import gzip
import pandas as pd
import sys

def proto_to_regex(proto):
    '''
    :param proto: string of input protospacer
    :return: regex form of the protospacer
    '''
    # dictionary containing equivalent nucleotides for each IUPAC ambiguous code
    amb_codes = {'N': '[AGCT]', 'V': '[AGC]', 'B': '[TCG]', 'H': '[ACT]', \
                 'D': '[AGT]', 'M': '[AC]', 'R': '[AG]', 'W': '[AT]', \
                 'S': '[CG]', 'Y': '[CT]', 'K': '[GT]'}

    # replaces ambiguous codes with values from the amb_codes dictionary
    for code, regex in amb_codes.items():
        proto = proto.replace(code, regex)

    return proto

def proto_to_complement(proto):
    '''
    finds complement sequence of protospacer
    :param proto: string of input protospacer
    :return: complement of the protospacer
    '''
    # dictionary containing complements of nucleotides and ambiguous codes
    amb_complement = {'N': 'N', 'V': 'B', 'B': 'V', 'H': 'D', 'D': 'H', \
                      'M': 'K', 'K': 'M', 'R': 'Y', 'Y': 'R', 'S': 'W', \
                      'W': 'S', 'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

    # initialize variable for storing complement of protospacer
    complement_protospacer = ''

    # iterates through protospacer sequence
    # finds complement of each nucleotide including ambigous codes
    for nucleotide in proto:
        if nucleotide in amb_complement.keys():
            nucleotide = amb_complement[nucleotide]
            complement_protospacer += nucleotide

    return complement_protospacer

def extractInfo(seq_name, seq, proto, length, target, len_proto, symbol, start_loc=0, end_loc=0):
    '''
    :param seq_name: sequence ID
    :param seq: sequence
    :param proto: protospacer
    :param length: length of the target site
    :param target: dictionary for storing target information
    :param len_proto: length of the protospacer
    :param symbol: + or -
    :param start_loc: 0 if no input, start index of the gene sequence
    :param end_loc: 0 if no input, end index of the gene sequence
    :return: target information in the format of {ID: Position, Protospacer, Sequence}
    '''

    # find all instances of given protospacer in the sequence and stores target information in the 'target' dictionary
    for match in re.finditer(proto, seq):
        pos = match.span() # contains start and end index of matched protospacer in a tuple
        target_pos = str(pos[0]+start_loc) +  '-' + str(pos[1]+end_loc+length) # concatenates start and end index of target+protospacer
        position = pos[0]+start_loc # protospacer start index
        protospacer = seq[pos[0]:pos[0]+len_proto] # sequence of protospacer
        ID_forward = seq_name + '_' + target_pos + '+' # concatenates ID with target position
        ID_backward = seq_name + '_' + target_pos + '-'

        if symbol == '-':
            ID = ID_backward
            sequence = seq[pos[1]-length-len_proto:pos[1]-len_proto] # target sequence
        elif symbol == '+':
            ID = ID_forward
            sequence = seq[pos[1]:pos[1]+length] # target sequence

        # format of target dictionary is as follows: {ID: Position, Protospacer, Sequence}
        if len(sequence) == length:
            target[ID] = [position, protospacer, sequence, seq_name]

    return target

def gff_target_types(gfffile):
    '''
    NOT USED
    extracts all target types in the gff file
    :param gfffile: gff file name
    :return: list of target types
    '''

    # extracts all target types in the gff file and stores them in a list
    target_types = list()
    for entry in GFF.parse(gfffile, target_lines=1000):
        for f in entry.features:
            target_types.append(f.type)
    return target_types

def transcriptome(gff_file):
    '''
    extracts start and end positions of all coding sequences
    :param gff_file: opened gff file name
    :param target_region: type of target region
    :return: list containing start and end positions of the target region
    '''
    gene_locs = list()
    for entry in GFF.parse(gff_file, target_lines=1000):
        for f in entry.features:
            if str(f.type) == 'gene':
                gene_locs.append(re.findall(r'\d+', str(f.location)))

    return gene_locs

def target_gene(gff_file, gene_ID):
    '''
    extracts location of the specified gene ID
    :param gff_file: opened gff file name
    :param gene_ID:
    :return: location of the gene
    '''

    gene_loc = ''
    for entry in GFF.parse(gff_file, target_lines=1000):
        for f in entry.features:
            if f.qualifiers['ID'][0] == gene_ID:
                gene_loc = re.findall(r'\d+', str(f.location))

    return gene_loc

def main(args):
    '''
    loads sequence file in either .gz or .fasta format
    can handle multi-fasta files
    outputs either .txt or .csv file depending on user specification
    :param args: args[0] = filename, args[1] = spacer, args[2] = length, args[3] = dir-rep orientation, args[4] = output filename
    args[5] = gff file or NA, args[6] = target gene ID
    :return: (temporary) output file for targets identified and common guides (common.csv)
    Enter on the command prompt for testing: python TIRv03c.py SARSCOV4.fasta TTTV 20 5 sample2.txt GCF_009858895.2_ASM985889v3_genomic.gff.gz gene-GU280_gp11
    '''

    # assign all args into variable names
    file = args[0]
    proto = args[1] # PAM/PFS
    length = int(args[2]) #20 bp
    orientation = int(args[3]) # 5 or 3
    #output_filename = args[4]
    gff = args[5] # gff file name
    gene_ID = args[6] # target gene ID
    chrom_id = args[7] # target chromosome ID


    # opens gff file if gff filename was entered
    if (gff != None) & (gene_ID != None):
        gff_file = gzip.open(gff, "rt")
        if gene_ID == 'All':
            transcriptome_locs = transcriptome(gff_file)
        else:
            gene_loc = target_gene(gff_file, gene_ID)

    # contains the format identifier of the file
    file_format = file.split('.')[-1]

    # initialize variables for storing
    target = dict()
    seq_names = list()
    merge_output = dict()
    len_proto = len(proto)

    # convert proto to regex and compile
    proto_regex_5 = proto_to_regex(proto)
    proto_regex_3 = proto_to_regex(proto[::-1])

    # convert proto to its reverse complement and compile
    comp_proto = proto_to_complement(proto)
    rev_proto = proto_to_complement(proto)[::-1]
    comp_regex = proto_to_regex(comp_proto)
    rev_proto_regex = proto_to_regex(rev_proto)

    # assigns protospacer location depending on direct repeat orientation
    if orientation == 5:
        merge_proto = [re.compile(proto_regex_5), re.compile(rev_proto_regex)]
    elif orientation == 3:
        merge_proto = [re.compile(comp_regex), re.compile(proto_regex_3)]

    # opens gz file
    if file_format == 'gz':
        file = gzip.open(file, "rt")

    # parses through file
    for seq_record in SeqIO.parse(file, "fasta"):
        seq_name, seq = seq_record.id, str(seq_record.seq)

        for motif, symbol in zip(merge_proto, ['+','-']):
            # if no gff file and gene ID have been provided, whole sequence will be searched
            if (gff == None) & (gene_ID == None) & (chrom_id == None):
                merge_output.update(extractInfo(seq_name, seq, motif, length, target, len_proto, symbol))

            # if gff and gene ID were provided, only specified region will be searched
            elif (gff != None) & (gene_ID != None):
                    start_locs = int(gene_loc[0])  # start index of gene
                    end_locs = int(gene_loc[1])  # end index of gene
                    seq_gene = seq[start_locs:end_locs]

                    # if chromosome not specified
                    if chrom_id == None:
                        merge_output.update(extractInfo(seq_name, seq_gene, motif, length, target, len_proto, symbol,
                                                        start_locs, end_locs))

                    # if chromosomes are specified
                    elif chrom_id != None:
                        if chrom_id == seq_name:
                            merge_output.update(extractInfo(chrom_id, seq_gene, motif, length, target, len_proto, symbol,
                                                            start_locs, end_locs))

    # converts target dictionary to pandas dataframe
    df_target = pd.DataFrame.from_dict(merge_output, orient='index').reset_index()
    df_target.columns = ['Target ID', 'Position', 'Protospacer', 'Sequence', 'Sequence ID']
    del merge_output


    # # outputs targets identified into either txt or csv
    # if output_filename.split('.')[-1] == 'txt':
    #     df_target.to_csv(output_filename, sep='\t', encoding='utf-8', index=False)
    # elif output_filename.split('.')[-1] == 'csv':
    #     df_target.to_csv(output_filename, encoding='utf-8', index=False)

    return df_target

if __name__ == "__main__":
   main(sys.argv[1:])