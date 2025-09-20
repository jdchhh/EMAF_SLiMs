This project implements a multi-modal protein classifier that integrates features from ProtBERT embeddings, PSSM (Position-Specific Scoring Matrix), and physicochemical properties using attention mechanisms and a Multi-Layer Perceptron (MLP). The model is trained and tested on labeled protein sequences for binary classification tasks. You can access the complete dataset used in this project via the following link: https://www.kaggle.com/datasets/duochenjin/emaf-slims-dataset/data.

\# EMAF\_SLiMs: Multi-Modal Protein Classifier for SLiMs Detection



This project implements a multi-modal protein classifier designed to detect Short Linear Motifs (SLiMs) in protein sequences. The model integrates features from ProtBERT embeddings, Position-Specific Scoring Matrices (PSSM), and physicochemical properties, leveraging attention mechanisms and a Multi-Layer Perceptron (MLP) for binary classification.





1\. Project Structure

| File/Directory       | Description                                                                 |

|----------------------|-----------------------------------------------------------------------------|

| `model.py`           | Defines the multi-modal classification model with attention and MLP layers. |

| `processed.py`       | Preprocessing script to convert raw protein sequences into model-compatible features. |

| `train.py`           | Trains the model on labeled data and handles hyperparameter tuning.         |

| `test.py`            | Evaluates the trained model on test data and generates performance metrics/visualizations. |

| `README.md`          | This documentation file.                                                     |

| `data/`              | (Not included in repo) Directory for raw/trained data; use the dataset link below for access. |





2\. Dataset Access

The complete dataset (including preprocessed features and raw sequences) can be accessed via:  

\[https://www.kaggle.com/datasets/duochenjin/emaf-slims-dataset/data](https://www.kaggle.com/datasets/duochenjin/emaf-slims-dataset/data)  





3\. Workflow for Single Sequence Analysis

If you want to test a single protein sequence with our model, follow these steps:



Step 1: Preprocess the Raw Sequence

Use `processed.py` to convert the raw sequence into feature vectors (ProtBERT embeddings, PSSM, and physicochemical properties).  

```bash

python processed.py --input your\_sequence.fasta --output preprocessed\_features.npy

```



Step 2: Test with Preprocessed Data

We provide preprocessed data in the dataset (linked above). To evaluate the model on test data (or your preprocessed single sequence), run:  

```bash

python test.py --test\_data preprocessed\_test\_data.npy --model\_path trained\_model.pth

```





4\. Model Training (Optional)

If you wish to retrain the model, use `train.py` with the preprocessed training data:  

```bash

python train.py --train\_data preprocessed\_train\_data.npy --val\_data preprocessed\_val\_data.npy

```





This workflow ensures you can either use our preprocessed data for immediate testing or process your own sequences for custom analysis.

