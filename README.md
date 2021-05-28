Biological activity prediction
=============
Source code of the Project for the Deep Learning course at Skoltech 2021.


This project is dedicated to biological activity prediction using deep learning (DL) architectures. 
We use the database of antiviral activity molecular data to create datasets. 
We employ the variety of ways to represent the molecules: feature matrix, 
molecular descriptors, molecular fingerprints, 
molecular graphs. We explore and compare the corresponding DL architectures: 
Attention-based Convolutional Neural Network (CNN), gradient boosting on decision trees + 
Multi-Layer Perceptron, transformer-based architectures and graph convolution
network (GCN) with two different convolution operators. 
As a result, we compared all models for binary and multi-class classification of biological activity, 
where the graph convolution network outperforms other deep learning methods for our datasets.
### Team members

* Anastasia Sarycheva
* Daria Chaplygina
* Alexey Voskoboinikov
* Roman Bychkov
* Sayan Protasov

### Brief repository overview

* `/notebooks` - the main directory where all jupyter notebooks with models are stored
   * `/datasets` - subdirectory where files associated with the dataset are stored
   * `/tune` - subdirectory where files associated with the baseline fine-tuning are stored
   * `/pretrained_transformer` - subdirectory where files associated with the transformer model are stored
   

### Notebooks
The following Google Colab compatible Jupyter notebooks with models are available:

* `Model_1D_CNN.ipynb` - CNN with 1D convolutional operator
* `Transformer.ipynb` - transformer-based architecture 
* `catboost4antiviral+DNN_v2.ipynb` - gradient boosting on decision trees
* `graph_conv_gcnconv.ipynb` - graph convolutional network with GCNConv operator
* `graph_conv_mfconv.ipynb` - graph convolutional network with MFConv operator
* `run_Pham2019.ipynb` - [baseline notebook](https://github.com/lehgtrung/egfr-att "Named link title")
* `run_Pham2019_v2_ourDS.ipynb` - baseline notebook on our datasets

### Datasets
All the datasets can be downloaded from here:
https://drive.google.com/drive/folders/1xpdrOMRxinYNCcNohQWPE3T-2La5bXOB?usp=sharing

### Requirements

The code is set up so that it can be easily run with Google Colab.
