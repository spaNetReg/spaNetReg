#!/bin/bash
# ===============================================================
# data_preprocess.sh
# ---------------------------------------------------------------
# Preprocessing pipeline for spaNetReg: generates the initial skeleton network
# and the feature matrix (regulatory potential, RP score)
# from raw ATAC-seq peak count matrices, following the design of DeepTFni.
#
# Usage:
#   bash data_preprocess.sh <sample_name> <reference>
#
# Example:
#   bash data_preprocess.sh Sample1 GRCm38
#
# Inputs:
#   <sample>/<sample>.csv      - ATAC-seq peak-by-cell matrix
#
# Outputs:
#   <sample>_temp/             - Temporary folder containing intermediate files
#   <sample>_TF_rp_score.txt   - TF-by-spot regulatory potential matrix (features)
#   <sample>_TF_list.txt       - List of transcription factors included in the skeleton network
#   <sample>.txt               - Binary adjacency matrix of the TF network
#
# Dependencies:
#   bedtools, bedops, sort-bed, FIMO, Python 3, MAESTRO
# ===============================================================
sample=$1
reference=$2

workdir=$(pwd)
outdir="${workdir}/${sample}_temp"
mkdir -p "$outdir"

if [[ "$reference" == "GRCm38" ]]; then
    genome_fa="${workdir}/data_resource/mm10.fa"
    motif_list="${workdir}/data_resource/mouse_HOCOMO/motif_mouse_list.txt"
    motif_db="${workdir}/data_resource/mouse_HOCOMO/HOCOMOCOv11_core_MOUSE_mono_meme_format.meme"
    promoter_file="${workdir}/data_resource/gencode.mm10.ProteinCoding_gene_promoter.txt"
    promoter_temp="$outdir/gencode.mm10.TF.promoter.temp"
    promoter_bed="$outdir/gencode.mm10.TF.promoter.txt"
elif [[ "$reference" == "GRCh38" ]]; then
    genome_fa="${workdir}/data_resource/hg38.fa"
    motif_list="${workdir}/data_resource/human_HOCOMO/motif_human_list.txt"
    motif_db="${workdir}/data_resource/human_HOCOMO/HOCOMOCOv11_core_HUMAN_mono_meme_format.meme"
    promoter_file="${workdir}/data_resource/gencode.hg38.ProteinCoding_gene_promoter.txt"
    promoter_temp="$outdir/gencode.hg38.TF.promoter.temp"
    promoter_bed="$outdir/gencode.hg38.TF.promoter.txt"
else
    echo "Unsupported reference: $reference" >&2
    exit 1
fi

infile="${workdir}/${sample}/${sample}.csv"
bed_tmp="$outdir/${sample}.clean.ATAC_peak.temp"
matrix_tmp="$outdir/${sample}.clean.ATAC_peak.matrix.temp"
bed_final="$outdir/${sample}.clean.ATAC_peak.bed"
matrix_final="$outdir/${sample}.clean.ATAC_peak.matrix.txt"
filtered_bed="$outdir/${sample}.filtered.bed"
fasta_output="$outdir/${sample}.clean.fasta"
fasta_final="$outdir/${sample}.fasta"
promoter_final="$outdir/${sample}_tfbs_filtered.txt"
fimo_out="$outdir/${sample}_1e4"



#  Filter - Keep peaks detected in >10% of cells
tail -n +2 "$infile" | awk -F',' -v OFS='\t' '
{
    gsub(/"/, "", $1);
    split($1, loc, /[:_\t-]/)
    chr = loc[1]; start = loc[2]; end = loc[3]
    n = 0
    for (i = 2; i <= NF; i++) if ($i > 0) n++
    ratio = n / (NF - 1)
    if (ratio > 0.1) {
        print chr, start-1, end > "'"$bed_tmp"'"
        printf "%s\t%s\t%s", chr, start, end > "'"$matrix_tmp"'"
        for (i = 2; i <= NF; i++) printf "\t%s", $i >> "'"$matrix_tmp"'"
        printf "\n" >> "'"$matrix_tmp"'"
    }
}
'
sort-bed "$bed_tmp" > "$bed_final"
sort-bed "$matrix_tmp" > "$matrix_final"
rm "$bed_tmp" "$matrix_tmp"
fimo_tsv="${fimo_out}/fimo.tsv"
tfbs_temp="${outdir}/${sample}.temp"
tfbs_final="${outdir}/${sample}.txt"

grep -v -P "\tchrY\t" "$bed_final" > "$filtered_bed"


# Extract FASTA sequences for filtered peaks
bedtools getfasta -fi "$genome_fa" -bed "$filtered_bed" -fo "$fasta_output"
# Adjust FASTA header formatting for FIMO
awk '/^>/{ 
    gsub(/^>/, "", $0);
    split($0, a, /[:-]/);
    new_start = a[2] + 1;
    print ">" a[1] ":" new_start "-" a[3];
    next;
}
{ print toupper($0)}
' "$fasta_output" > "$fasta_final"


# Motif scanning using FIMO
fimo --oc "$fimo_out" --no-qvalue "$motif_db" "$fasta_final"

# Extract TFBS from FIMO results (p-value ≤ 1e-6)
awk -F'\t' 'BEGIN{OFS="\t"}
    NF > 0 && $0 !~ /^#/ && $1 !~ /motif_id/ && $8 <= 1e-6 {
        split($1, tfparts, "_");
        tf = tfparts[1];

        split($3, loc, /[:\-]/);
        chr = loc[1];
        start = loc[2] + $4 - 1;
        end = loc[2] + $5 - 1;
        print chr, start, end, tf
    }' "$fimo_tsv" > "$tfbs_temp"

sort-bed "$tfbs_temp" > "$tfbs_final"
rm "$tfbs_temp"

echo "TFBS region extraction completed: $tfbs_final"


awk 'NR==FNR {motif[$1]=1; next} ($NF in motif)' "$motif_list" "$promoter_file" > "$promoter_temp"
sort-bed "$promoter_temp" > "$promoter_bed"
rm "$promoter_temp"

bedops -e -100% "$tfbs_final" "$promoter_bed" > "$promoter_final"

# Construct TF–gene adjacency matrix and MAESTRO inputs
python ${workdir}/preprocess/build_adj.py --workdir "${workdir}" --sample "${sample}" --ref $reference 
python ${workdir}preprocess/built_MAESTRO_input.py -sample "${sample}" -input_file $infile -output_dir $outdir -ref $$reference

# Run MAESTRO to compute gene activity scores
MAESTRO scatac-genescore --format h5 --peakcount ${outdir}/${sample}_peak_count.h5 --genedistance 10000 --species $$reference --model Enhanced --outprefix ${outdir}/${sample}
python ${workdir}/preprocess/TF_rp_score.py $sample $workdir
