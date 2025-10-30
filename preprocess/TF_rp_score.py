import sys
import os
import pandas as pd
import h5py
from scipy import sparse

def main(sample, wd):
    """extracts TF RP score matrices from MAESTRO-generated HDF5 files."""
    wd = wd
    tf_path = os.path.join(wd, sample, f"{sample}_TF_list.txt")
    h5_path = os.path.join(wd, f"{sample}_temp", f"{sample}_gene_score.h5")
    out_path = os.path.join(wd, sample, f"{sample}_TF_rp_score.txt")
    tf_list = pd.read_csv(tf_path, header=None)[0].str.upper().tolist()

    with h5py.File(h5_path, "r") as f:
        genes = [g.decode() for g in f['matrix']['features']['name'][:]]
        barcodes = [b.decode() for b in f['matrix']['barcodes'][:]]
        data = f['matrix']['data'][:]
        indices = f['matrix']['indices'][:]
        indptr = f['matrix']['indptr'][:]
        shape = f['matrix']['shape'][:]

    matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
    gene_upper = [g.upper() for g in genes]

    gene_df = pd.DataFrame.sparse.from_spmatrix(matrix, index=gene_upper, columns=barcodes)

    tf_exist = [tf for tf in tf_list if tf in gene_df.index]
    tf_matrix = gene_df.loc[tf_exist]


    tf_matrix.to_csv(out_path, sep="\t", index=False)

    print(f"✔ Done! TF RP score saved to: {out_path}")
    print(f"✔ TFs used: {len(tf_exist)} / {len(tf_list)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_tf_rp_score.py <sample> <work_dir>")
        sys.exit(1)
    sample = sys.argv[1]
    wd = sys.argv[2]
    main(sample, wd)
