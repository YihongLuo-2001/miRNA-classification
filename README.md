# README

> We have run the model on CentOS Linux release 7.9.2009.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Usage](#usage)
- [Contact](#contact)
## Introduction

This site provides the information associated with the following paper: 

> Comparison of five machine learning models on the effectiveness of plant and animal miRNA classification



## Project Structure

```shell
miRNA-classification/
├── bin
│   ├── animal_plant  # Binary Classification
│   │   ├── cnn_test.py  # Testing Convolutional Neural Networks(CNN) Model
│   │   ├── cnn_train.py  # Training Convolutional Neural Networks(CNN) Model
│   │   ├── d_tree.py  # Training and testing Decision Tree(DT) Model
│   │   ├── nn_test.py  # Testing Neural Networks(NN) Model
│   │   ├── nn_train.py  # Training Neural Networks(NN) Model
│   │   ├── rf2.py  # Training and testing Random Forest(RF) Model
│   │   └── SVM_4.py  # Training and testing Support Vector Machine(SVM) Model
│   ├── data_processing
│   │   ├── binary_one-hot.py  # One-hot encoding(Binary Classification)
│   │   ├── dataset_split.py  # Split the dataset into training set and test set
│   │   ├── families_filter_0.py #  Filter data
│   │   ├── families_filter_1.py #  Filter data
│   │   ├── families_making_dic.py # Organize data labels
│   │   └── families_one-hot.py  # One-hot encoding(miRNA Families Classification)
│   ├── explanation  # Analyze Random Forest Model
│   │   ├── make_svg.sh  # Convert the dot file into a visualized SVG file.
│   │   ├── random_forests.py  # Compute feature importances, analyze the path in the model and convert the model into dot files
│   │   └── svm_hyperplane_function.py  # Print hyperplane function of SVM model
│   └── families_20  # miRNA Families Classification
│       ├── cnn_test_families.py  # Testing Convolutional Neural Networks(CNN) Model
│       ├── cnn_train_families.py # Training Convolutional Neural Networks(CNN) Model
│       ├── d_tree.py  # Training and testing Decision Tree(DT) Model
│       ├── nn_test_families.py  # Testing Neural Networks(NN) Model
│       ├── nn_train_families.py  # Training Neural Networks(NN) Model
│       ├── rf2.py  # Training and testing Random Forest(RF) Model
│       └── SVM_4.py  # Training and testing Support Vector Machine(SVM) Model
├── checking_environment.py  # Checking PyTorch and sk-learn packages
├── data
│   ├── animal_plant_dataset  # Dataset(Binary Classification)
│   │   ├── Animal_miRNA.txt  # Animal miRNA Raw data
│   │   ├── one-hot.txt  # One-hot encoded data
│   │   ├── Plant_miRNA.txt  # Plant miRNA Raw data
│   │   ├── rna_test.txt  # Testing set
│   │   └── rna_train.txt  # Training set
│   └── families_20_dataset  # Dataset(miRNA Families Classification)
│       ├── Animal_miRNA.txt # Animal miRNA Raw data
│       ├── one-hot-families.txt # Raw one-hot data
│       ├── Plant_miRNA.txt  # Plant miRNA Raw data
│       ├── set-families_count.csv  # Data label reference table
│       ├── set-families_dic.csv  # Data label reference table
│       ├── set-one-hot-families-filtered.txt  # Filtered one-hot data
│       ├── set-one-hot-families.txt  # Raw one-hot data
│       ├── set_test_families-filtered.txt  # Testing set
│       └── set_train_families-filtered.txt  # Training set
├── models_results
│   ├── animal_plant  # models and results(Binary Classification)
│   │   ├── best_decision_tree_model.joblib  # Decision Tree(DT) Model file
│   │   ├── best_random_forest_model_5.z01  # A part of Random Forest(RF) Model file
│   │   ├── best_random_forest_model_5.z02  # A part of Random Forest(RF) Model file
│   │   ├── best_random_forest_model_5.z03  # A part of Random Forest(RF) Model file
│   │   ├── best_random_forest_model_5.zip  # A part of Random Forest(RF) Model file
│   │   ├── model_cnn.pkl  # Convolutional Neural Networks(CNN) Model file
│   │   ├── cnn_result.txt  # Convolutional Neural Networks(CNN) Model testing results
│   │   ├── decision_tree_results.txt  # Decision Tree(DT) Model testing results
│   │   ├── model_nn.pkl  # Neural Networks(NN) Model file
│   │   ├── nn_result.txt  # Neural Networks(NN) Model testing results
│   │   ├── random_forest_results.txt  # Random Forest(RF) Model testing results
│   │   ├── svm_model_6.joblib  # Support Vector Machine(SVM) Model file
│   │   └── SVM_report_6.txt  # Support Vector Machine(SVM) Model testing results
│   ├── explanation  # models and results(Analyze Random Forest Model)
│   │   ├── hyperplane.txt  # Hyperplane function of the SVM model
│   │   ├── position.tsv  # Sum the feature importance of all decision criteria.
│   │   ├── trees_dot/  # dot files of the Random Forest model 
│   │   ├── trees_svg/  # Visualized SVG file
│   │   └── trees.tsv  # The feature importance of each node in the 1-4 layers. 
│   └── families_20  # models and results(miRNA Families Classification)
│       ├── best_decision_tree_model_families.joblib  # Decision Tree(DT) Model file
│       ├── best_random_forest_model_families.zip  # Random Forest(RF) Model file
│       ├── cnn_result.txt  # Convolutional Neural Networks(CNN) Model testing results
│       ├── decision_tree_results_families.txt  # Decision Tree(DT) Model testing results
│       ├── model_cnn_families.pkl  # Convolutional Neural Networks(CNN) Model file
│       ├── model_nn_families.pkl  # Neural Networks(NN) Model file
│       ├── nn_result.txt  # Neural Networks(NN) Model testing results
│       ├── random_forest_results_families.txt  # Random Forest(RF) Model testing results
│       ├── svm_model_6_families.joblib  # Support Vector Machine(SVM) Model file
│       └── SVM_report_6_families.txt  # Support Vector Machine(SVM) Model testing results
└── README.md  # README file

```



## Environment

Create a conda environment

```shell
conda create --name mirna python=3.8 -y
conda activate mirna

```



Use conda to install.

```shell
conda install -c conda-forge git -y
conda install -c conda-forge graphviz -y
conda install -c conda-forge unzip -y

```



Use pip to install.

```shell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn==1.3.2

```



Check CUDA version and graphics cards

> A graphics card with more than 4096 MB of memory is required.
>
> Taking NVIDIA graphics cards as an example.

```shell
nvidia-smi

```

e.g. The  CUDA Version is 11.4, memory is 24268 MB

```shell
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:4B:00.0 Off |                  N/A |
| 39%   29C    P8    29W / 350W |      0MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:CA:00.0 Off |                  N/A |
| 41%   26C    P8    20W / 350W |      0MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```



Install PyTorch according to the CUDA version

> Visit the URL 'https://download.pytorch.org/whl/torch_stable.html' to find the appropriate package.

```shell
# pip install torch==X.XX.X+cuXXX -f https://download.pytorch.org/whl/torch_stable.html
# pip install torchvision==X.XX.X+cuXXX -f https://download.pytorch.org/whl/torch_stable.html
# e.g. If your CUDA version ≥ 11.3, you can execute the following command line.
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

```



Git clone

```shell
git clone https://github.com/YihongLuo-2001/miRNA-classification.git
# It may take a lot of time. You can also download it manually

```
 Manually download (If "Git clone" doesn't work)

```shell
wget https://github.com/YihongLuo-2001/miRNA-classification/archive/refs/heads/main.zip
unzip main.zip
mv miRNA-classification-main/ miRNA-classification/

```



Check python package

```shell
cd miRNA-classification/
python checking_environment.py  # check pytorch and sk-learn

```



Unzip large file

```shell
cd models_results/animal_plant/
cat best_random_forest_model_5.z* > best_random_forest_model_5_total.zip
unzip best_random_forest_model_5_total.zip
md5sum best_random_forest_model_5.joblib  # md5: d02c54b7b7b155dc96789aeac7abee2b
cd ../families_20/
unzip best_random_forest_model_families.zip
md5sum best_random_forest_model_families.joblib  # md5: 10af813c1452be3d1f2bdf4f24c1943f
cd ../..

```



## Usage

Dataset Processing

```shell
cd bin/data_processing/
python binary_one-hot.py
python families_one-hot.py 
python families_filter_0.py
python families_making_dic.py 
python families_filter_1.py
python dataset_split.py 
cd ../..

```



miRNA Families Classification Model training

```shell
cd bin/families_20/
python d_tree.py
python cnn_train_families.py
python cnn_test_families.py > ../../models_results/families_20/cnn_result.txt
python nn_train_families.py
python nn_test_families.py > ../../models_results/families_20/nn_result.txt
python rf2.py  # You can adjust the threads in the code manually (Default 20 threads).
python SVM_4.py
cd ../..

```



View miRNA Families Classification results

```shell
cat models_results/families_20/*txt

```



Binary Classification Model training

```shell
cd bin/animal_plant/
python cnn_train.py
python cnn_test.py > ../../models_results/animal_plant/cnn_result.txt
python nn_train.py
python nn_test.py > ../../models_results/animal_plant/nn_result.txt
python d_tree.py  # You can adjust the threads in the code manually (Default 20 threads).
python rf2.py  # You can adjust the threads in the code manually (Default 20 threads).
python SVM_4.py  # It may take a lot of time (About 2-4 hours). You can adjust the threads in the code manually (Default 20 threads).
cd ../..

```



View Binary Classification results

```shell
cat models_results/animal_plant/*txt
```



Analyze Random Forest Model

```shell
cd bin/explanation/
python svm_hyperplane_function.py > ../../models_results/explanation/hyperplane.txt
cat ../../models_results/explanation/hyperplane.txt
python random_forests.py 
bash make_svg.sh
cd ../../models_results/explanation/trees_svg/  # Svg files

```



## Contact

Main developer: Yihong Luo

> E-mail: luoyihong2001@163.com



