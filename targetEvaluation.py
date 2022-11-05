import collections
import math
import re
import pandas as pd
import targetMFE


def GC_content(subseq):
    gc_count = 0
    for i in subseq.upper():
        if i == "G" or i == "C":
            gc_count += 1

    return gc_count / len(subseq)


def entropy(subseq):
    m = len(subseq)
    bases = collections.Counter([tmp_base for tmp_base in subseq])

    entropy = 0
    for base in bases:
        n_i = bases[base]
        p_i = n_i / float(m)
        entropy_i = p_i * (math.log(p_i, 2))
        entropy += entropy_i

    return (entropy * -1) / 2


def tandem_repeat(seq):
    try:
        result = max(re.findall(r'((\w+?)\2+)', seq), key=lambda t: len(t[0]))
        return len(result[0]) / len(seq)
    except:
        return 0


def melting_temperature(seq):
    seq_list = list(seq)
    if len(seq_list) < 14:
        Tm = (2 * (seq_list.count('A') + seq_list.count('T'))) + (4 * (seq_list.count('C') + seq_list.count('G'))) # divide by length of sequence!

    elif len(seq_list) > 13:
        Tm = 64.9 + 41 * (seq_list.count('C') + seq_list.count('G') - 16.4) / (
                    seq_list.count('A') + seq_list.count('T') + seq_list.count('C') + seq_list.count('G'))

    return Tm


def contiguous_repeat(seq):
    # adapted from geeksforgeeks.com
    ans, temp = 1, 1

    # Traverse the string
    for i in range(1, len(seq)):

        # If character is same as
        # previous increment temp value
        if (seq[i] == seq[i - 1]):
            temp += 1
        else:
            ans = max(ans, temp)
            temp = 1

    ans = max(ans, temp)

    # Return the required answer
    return ans / len(seq)


def norm_melting_temperature(seq):
    seq_list = list(seq)
    if len(seq_list) < 14:
        Tm = (2 * (seq_list.count('A') + seq_list.count('T'))) + (4 * (seq_list.count('C') + seq_list.count('G')))
        norm_Tm = Tm / (2 * (0) + (4 * len(seq)))
    elif len(seq_list) > 13:
        Tm = 64.9 + 41 * (seq_list.count('C') + seq_list.count('G') - 16.4) / (
                    seq_list.count('A') + seq_list.count('T') + seq_list.count('C') + seq_list.count('G'))
        norm_Tm = Tm / (64.9 + 41 * ((len(seq) - 16.4) / len(seq)))

    return norm_Tm


def nucleotide_freq(seq):

    seq = list(seq)
    A = (seq.count('A') / len(seq))
    C = (seq.count('C') / len(seq))
    T = (seq.count('T') / len(seq))
    G = (seq.count('G') / len(seq))

    return [A, C, T, G]


def MFE(df_common):
    # MFE_filename = targetMFE.seq2fasta([df_common], ["Common"])

    # LINE BELOW IS COMMENTED TO SAVE PROCESSING TIME. ONLY UNCOMMENT IF YOU WANT TO START PROCESSING NEW INPUT FILES
    # targetMFE.target_MFE(MFE_filename, "common_output_mfe.txt")
    mfe = [float(i) for i in targetMFE.mfe2df("common_output_MFE.txt")]

    return mfe


def pos_spec_nucl(df_common):
    pos = dict()
    nucleotides = list('AAAAGGGGTTTTCCCC')
    position = [i for i in range(1, 29)]

    for i in position:
        for j in nucleotides:
            pos.update({'Pos_' + str(i) + '_' + str(j): [0 for k in range(len(df_common['Sequence']))]})

    for idx, i in enumerate(df_common['Sequence']):
        seq = list(i)
        for j in range(len(seq)):
            if seq[j] == 'A':
                pos['Pos_' + str(j + 1) + '_' + 'A'][idx] = 1
            elif seq[j] == 'T':
                pos['Pos_' + str(j + 1) + '_' + 'T'][idx] = 1
            elif seq[j] == 'C':
                pos['Pos_' + str(j + 1) + '_' + 'C'][idx] = 1
            elif seq[j] == 'G':
                pos['Pos_' + str(j + 1) + '_' + 'G'][idx] = 1

    df_pos = pd.DataFrame(pos)

    return df_pos


def target_features(df_common, enzyme):

    target = dict()

    cas12 = [1 if enzyme == '12' else 0 for i in range(len(df_common))]
    cas13 = [1 if enzyme == '13' else 0 for i in range(len(df_common))]
    cas9 = [1 if enzyme == '9' else 0 for i in range(len(df_common))]

    mfe_lst = MFE(df_common)

    for seq, mfe, c12, c13, c9 in zip(df_common['Sequence'], mfe_lst, cas12, cas13, cas9):
        gc_content = GC_content(seq)
        tan_rep = tandem_repeat(seq)
        ent = entropy(seq)
        n_melt_temp = norm_melting_temperature(seq)
        cont_rep = contiguous_repeat(seq)
        A_count = nucleotide_freq(seq)[0]
        C_count = nucleotide_freq(seq)[1]
        T_count = nucleotide_freq(seq)[2]
        G_count = nucleotide_freq(seq)[3]

        target[seq] = [gc_content, tan_rep, ent, n_melt_temp, cont_rep, mfe, c12, c13, c9, A_count, C_count, T_count, G_count]

    df_target_ft = pd.DataFrame.from_dict(target, orient='index').reset_index()
    df_target_ft.columns = ['guide_seq', 'GC_content', 'tandem_repeats', 'entropy', 'melting_temperature', 'contiguous_repeats', 'MFE', \
                            'Cas12', 'Cas13', 'Cas9','A_count','C_count', 'T_count', 'G_count']

    df_target_ft['MFE'] = (df_target_ft['MFE'] - df_target_ft['MFE'].min()) / (
                df_target_ft['MFE'].max() - df_target_ft['MFE'].min())

    df_pos = pos_spec_nucl(df_common)   # extracts position specific nucleotide features

    df_target_fin = pd.concat([df_target_ft, df_pos], axis=1)

    return df_target_fin





