import pandas as pd
import numpy as np
import os
import h5py
import numpy
import numpy as np
import scipy.io
import gzip
import scipy.sparse as sp_sparse
import pandas as pd
from scipy.sparse import coo_matrix



# Preprocessing module similar to DeepTFni:
# converts ATAC-seq peak-by-cell count matrices (CSV) into 10X-style
# HDF5 format compatible with MAESTRO for downstream RP score calculation.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-sample", "--sample_type", type=str, help="the name of input")
parser.add_argument("-input_file", "--input_dir_file", type=str, help="the path of input")
parser.add_argument("-output_dir", "--output_directory", type=str, help="the path of h5 file output")
parser.add_argument("-ref", "--ref", type=str)

args = parser.parse_args()


def write_10X_h5(filename, matrix, features, barcodes, genome='GRCh38', datatype='Peak'):
    """Write 10X HDF5 files, support both gene expression and peaks."""

    f = h5py.File(filename, 'w')

    if datatype == 'Peak':

        M = sp_sparse.csc_matrix(matrix, dtype=numpy.int8)

    else:

        M = sp_sparse.csc_matrix(matrix, dtype=numpy.float32)

    B = numpy.array(barcodes, dtype='|S200')

    P = numpy.array(features, dtype='|S100')

    GM = numpy.array([genome] * len(features), dtype='|S10')

    FT = numpy.array([datatype] * len(features), dtype='|S100')

    AT = numpy.array(['genome'], dtype='|S10')

    mat = f.create_group('matrix')

    mat.create_dataset('barcodes', data=B)

    mat.create_dataset('data', data=M.data)

    mat.create_dataset('indices', data=M.indices)

    mat.create_dataset('indptr', data=M.indptr)

    mat.create_dataset('shape', data=M.shape)

    fet = mat.create_group('features')

    fet.create_dataset('_all_tag_keys', data=AT)

    fet.create_dataset('feature_type', data=FT)

    fet.create_dataset('genome', data=GM)

    fet.create_dataset('id', data=P)

    fet.create_dataset('name', data=P)

    f.close()


def read_10X_mtx(matrix_file, feature_file, barcode_file, datatype, gene_column=2):
    """Convert 10x mtx as matrix."""

    print('read_10X_mtx function info')

    print('\tprocessing feature_file')
    if feature_file.split('.')[-1] == 'gz' or feature_file.split('.')[-1] == 'gzip':

        feature_in = gzip.open(feature_file, "r")

    else:

        feature_in = open(feature_file, "r")

    features = feature_in.readlines()

    if datatype == "Peak":

        features = ["_".join(feature.strip().split("\t")[0:3]) for feature in features]

    else:

        if type(features[0]) == str:
            features = [feature.strip().split("\t")[gene_column - 1] for feature in features]

        if type(features[0]) == bytes:
            features = [feature.decode().strip().split("\t")[gene_column - 1] for feature in features]
    print('\texample of feature[0] : ' + str(features[0]))
    print('\n')

    print('\tprocessing barcode_file')
    if barcode_file.split('.')[-1] == 'gz' or barcode_file.split('.')[-1] == 'gzip':

        barcode_in = gzip.open(barcode_file, "r")

    else:

        barcode_in = open(barcode_file, "r")

    barcodes = barcode_in.readlines()

    if type(barcodes[0]) == str:
        barcodes = [barcode.strip().split("\t")[0] for barcode in barcodes]

    if type(barcodes[0]) == bytes:
        barcodes = [barcode.decode().strip().split("\t")[0] for barcode in barcodes]
    print('\texample of barcode[0] : ' + str(barcodes[0]))
    print('\n')

    print('\tprocessing matrix_file')
    matrix = scipy.io.mmread(matrix_file)

    matrix = sp_sparse.csc_matrix(matrix, dtype=numpy.float32)

    return {"matrix": matrix, "features": features, "barcodes": barcodes}


def mtx_2_h5(directory, outprefix, matrix_file, feature_file, barcode_file, gene_column=2, genome='GRCh38',
             datatype='Peak'):
    """Convert 10x mtx format matrix to HDF5."""

    try:

        os.makedirs(directory)

    except OSError:

        # either directory exists (then we can ignore) or it will fail in the

        # next step.

        pass

    if datatype == "Peak":

        filename = os.path.join(directory, outprefix + "_peak_count.h5")

    else:

        filename = os.path.join(directory, outprefix + "_gene_count.h5")

    print('mtx_2_h5 function info')
    print('\ttarget_file : ' + filename)
    print('\tmatrix_file : ' + matrix_file)
    print('\tfeature_file : ' + feature_file)
    print('\tbarcode_file : ' + barcode_file)
    print('\n')

    matrix_dict = read_10X_mtx(matrix_file=matrix_file, feature_file=feature_file, barcode_file=barcode_file,
                               datatype=datatype, gene_column=gene_column)

    write_10X_h5(filename=filename, matrix=matrix_dict["matrix"], features=matrix_dict["features"],
                 barcodes=matrix_dict["barcodes"], genome=genome, datatype=datatype)


sample_type = args.sample_type
input_dir_file = args.input_dir_file
output_directory = args.output_directory

tmp = pd.read_csv(input_dir_file, sep=',', index_col=0)
data = tmp.values

data_sparse = coo_matrix(data)
coordinate_with_count = np.array([data_sparse.row + 1, data_sparse.col + 1, data_sparse.data])
Num = len(data_sparse.data)
P_num = np.size(data, 0)
C_num = np.size(data, 1)

input_directory = output_directory
if not (os.path.exists(input_directory)):
    os.mkdir(input_directory)
mtx_file_dir = input_directory + "/mtx_file"
if not (os.path.exists(mtx_file_dir)):
    os.mkdir(mtx_file_dir)

np.savetxt(mtx_file_dir + "/feature.txt", tmp.axes[0], fmt='%s')
np.savetxt(mtx_file_dir + "/barcodes.txt", tmp.axes[1], fmt='%s')
print()

mtx_file = mtx_file_dir + "/" + sample_type + ".mtx"
if os.path.exists(mtx_file):
    os.remove(mtx_file)
with open(mtx_file, 'a') as f:
    f.write("%%MatrixMarket matrix coordinate integer general\n")
    f.write("%\n")
    f.write(str(P_num) + " " + str(C_num) + " " + str(Num) + "\n")
    np.savetxt(f, coordinate_with_count.transpose(), fmt='%d', delimiter=' ')

matrix_file = os.path.join(mtx_file_dir + "/" + sample_type + ".mtx")
feature_file = os.path.join(mtx_file_dir + "/feature.txt")
barcode_file = os.path.join(mtx_file_dir + "/barcodes.txt")

mtx_2_h5(directory=output_directory, outprefix=sample_type, matrix_file=matrix_file, feature_file=feature_file,
         barcode_file=barcode_file, gene_column=2, genome=args.ref,
         datatype='Peak')



