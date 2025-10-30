import argparse
from collections import defaultdict
import os
import pandas as pd

def load_promoters(promoter_bed):
    """Load promoter regions for each transcription factor."""
    promoter_dict = defaultdict(list)
    with open(promoter_bed) as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 5:
                continue
            chr_, start, end, _, tf = fields
            promoter_dict[tf].append((chr_, int(start), int(end)))
    return promoter_dict

def load_tfbs(tfbs_bed):
    """Load TF binding sites."""
    tfbs_list = []
    with open(tfbs_bed) as f:
        for line in f:
            fields = line.strip().split('\t')
            chr_, start, end, tf = fields
            tfbs_list.append((chr_, int(start), int(end), tf))
    return tfbs_list

def build_adjacency(promoter_dict, tfbs_list):
    """Construct adjacency matrix: For each TF_B with TFBS, if its binding region falls within the promoter region of TF_A, add an undirected edge A–B."""
    adj = defaultdict(lambda: defaultdict(int))
    all_tfs = set()
    for chr_b, start_b, end_b, tf_b in tfbs_list:
        for tf_a, regions in promoter_dict.items():
            for chr_a, start_a, end_a in regions:
                if chr_a == chr_b and start_b >= start_a and end_b <= end_a:
                    adj[tf_a][tf_b] = 1
                    adj[tf_b][tf_a] = 1
                    all_tfs.update([tf_a])
    return adj, sorted(all_tfs)

def write_output(adj, all_tfs, out_adj_path, out_tf_list_path):
    df = pd.DataFrame(0, index=all_tfs, columns=all_tfs)

    for tf1 in all_tfs:
        for tf2 in all_tfs:
            if adj[tf1].get(tf2, 0) == 1:
                df.loc[tf1, tf2] = 1
    df.to_csv(out_adj_path, sep='\t')

    with open(out_tf_list_path, 'w') as f:
        for tf in all_tfs:
            f.write(tf + "\n")
    
    return df

def print_density(df):
    n = df.shape[0]
    total = n * n
    connected = df.values.sum()
    density = connected / total if total > 0 else 0
    print(f"✔ TF number: {n}")
    print(f"✔ Adjacency matrix density: {int(connected)} / {total} = {density:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Build TF-TF adjacency matrix from promoter and TFBS")
    parser.add_argument('--workdir')
    parser.add_argument('--sample', required=True, help='Sample name (used to locate TFBS and write output)')
    parser.add_argument('--ref')
    args = parser.parse_args()
    workdir = args.workdir
    sample = args.sample

    
    if args.ref == 'GRCm38':
        promoter_bed = f"{workdir}/{sample}_temp/gencode.mm10.TF.promoter.txt"
    elif args.ref == 'GRCh38':
        promoter_bed = f"{workdir}/{sample}_temp/gencode.hg38.TF.promoter.txt"
    else:
        raise ValueError(f"Unsupported reference genome: {args.ref}")
    
    tfbs_bed = f"{workdir}/{sample}_temp/{sample}_tfbs_filtered.txt"
    out_adj = f"{workdir}/{sample}/{sample}.txt"
    out_tf_list = f"{workdir}/{sample}/{sample}_TF_list.txt"

    if not os.path.exists(promoter_bed):
        raise FileNotFoundError(f"Promoter file not found: {promoter_bed}")
    if not os.path.exists(tfbs_bed):
        raise FileNotFoundError(f"TFBS file not found: {tfbs_bed}")

    os.makedirs(f"{workdir}/{sample}", exist_ok=True)

    promoter_dict = load_promoters(promoter_bed)
    tfbs_list = load_tfbs(tfbs_bed)
    adj, all_tfs = build_adjacency(promoter_dict, tfbs_list)
    df = write_output(adj, all_tfs, out_adj, out_tf_list)
    print_density(df)

if __name__ == "__main__":
    main()
