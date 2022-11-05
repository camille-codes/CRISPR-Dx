import argparse
import targetEvaluationModel
import targetIdentification
import uniquenessEvaluation
import commonStrands
import targetEvaluation
import pandas as pd
import targetMFE

# Copy arguments below for an example use
# python3 masterScript.py --input SARSCOV2a.fasta SARSCOV2b.fasta --offtargets ebola.txt --motif TTTV --length 20 --orientation 5 --enzyme 12 --output output_ebola1.txt --fm /home/camilles/offtargets/offtargets
# /home/camilles/offtargets/offtargets --reference influenza_a.txt --targets fasta_high_RNA.txt > "+ args.output

def main():
    parser = argparse.ArgumentParser(description='Extract target sites from sequence')
    parser.add_argument('--input', nargs="*", help="Input Fasta files in .fasta or .gz", metavar='', required=True)
    parser.add_argument('--motif', help="PAM or PFS", required=True)
    parser.add_argument('--length', help="Length of the target site", required=True)
    parser.add_argument('--orientation', help="Direct repeat orientation (5 or 3)", required=True)
    parser.add_argument('--enzyme', help="Cas enzyme (9, 12, or 13", required=True)
    parser.add_argument('--output', help="Output target sites in .csv or .txt", metavar='', required=True)
    parser.add_argument('--offtargets', help="Off-target sequences in .csv or .txt", metavar='', required=False)
    parser.add_argument('--gff', help="Input GFF file", required=False, const=None)
    parser.add_argument('--gene', help="Input target gene ID", required=False, const=None)
    parser.add_argument('--chrom', nargs="*", help="Input target chromosome ID in each file", metavar='',
                        required=False, const=None, type=str)
    parser.add_argument('--fm', help="path directory to Fuzzy Matching", required=True)
    args = parser.parse_args()

    # MODULE 1: TARGET IDENTIFICATION-----------------------------------------------------------------------------------
    df_targets = pd.DataFrame()

    if args.chrom is None:
        args.chrom = [None for i in range(len(args.input))]

    for input_file, output_file, chrom_id in zip(args.input, args.output, args.chrom):
        targets = targetIdentification.main([input_file, args.motif, args.length, args.orientation, args.output, args.gff,
                                             args.gene, chrom_id])
        df_targets = df_targets.append(targets)

    # MODULE 2: TARGET EVALUATION---------------------------------------------------------------------------------------
    df_common = commonStrands.common_strand(df_targets, len(args.input))
    df_target_fin = targetEvaluation.target_features(df_common, args.enzyme)

    # calls the ML model to make a prediction. Adds label, either 0 or 1
    df_common_w_label = targetEvaluationModel.predict_ML(df_target_fin)

    # extract high activity guide-RNAs
    high_activity_RNA = df_common_w_label[df_common_w_label['label'] == 1]

    print(len(high_activity_RNA))

    # high_activity_RNA.to_csv('high_activity_RNA.txt', sep='\t', encoding='utf-8', index=False)

    # convert file to fasta format, where each guide-RNA sequence has its own individual ID
    #targetMFE.seq2fasta_common(high_activity_RNA, "high_activity_RNA")

    # MODULE 3: UNIQUENESS EVALUATION-----------------------------------------------------------------------------------
    uniquenessEvaluation.run_offtargets_default(args.fm + " --reference " + args.offtargets + " --targets fasta_high_RNA.txt --mismatches 20 --pam "+args.motif + " > " + args.output)




if __name__ == '__main__':
    main()
