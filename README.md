# spaNetReg
spaNetReg is a framework designed to infer transcription factor regulatory network(TRN) from spatial ATAC-seq data. By integrating spatial information through a Gaussian Process (GP) prior and graph structure learning via a Variational Graph Autoencoder (VGAE),  
spaNetReg achieves reconstruction of regulatory interactions in complex tissue architectures.

![](./data_resource/framework.svg)

*Figure: The overall framework of the spaNetReg.*


## Requirements
* python: 3.9.21
* pytorch: 1.11.0
* numpy: 1.26.0
* pandas: 2.2.2
* MAESTRO: 1.5.1
* R: 4.0.5
* meme: 5.4.1


## Running
### 1 Preparation Datasets
All datasets used in this study are publicly available. 

* [MISAR-seq](https://doi.org/10.1038/s41592-023-01884-1)
* [Human Hippocampus](https://doi.org/10.1038/s41586-023-05795-1)
* [Human Glioblastoma](https://hub.uu2025.xyz/10.1073/pnas.2424070122)

The expected working directory structure for each dataset is as follows:
```
<sample>/
├── <sample>.csv          # ATAC-seq peak-by-cell matrix
└── <sample>_pos.csv      # Spatial coordinates (x, y)
```


### 2 Preprocess the Data
Construction of initial TRN skeleton and node feature (regulatory potential scores) for model training.

```
 bash data_preprocess.sh sample reference #Bash
``` 
The reference argument specifies the genome annotation reference and currently supports two species: human (hg38) and mouse (mm10).
For example:
```
bash data_preprocess.sh sample mm10 #Bash
``` 

### 3 Run the Model
```
python run_spaNetReg --sample sample #Bash
``` 



