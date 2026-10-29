# Deep-Learning-Based-Detection-of-Fake-Images-in-Saudi-Arabia

## Project Overview
This project investigates deep learning methods for distinguishing real photographs from AI-generated images, focusing on security events, the Two Holy Mosques, and entertainment festivals in Saudi Arabia.

## Phase 1 : Data Foundations and Image Classification
Phase 1 covers dataset prepartion, image clasifcation, model comparison, and evaluation.

### Data Preparation (M1)
1. Dataset inspection and label verification
2. Related-image grouping to reduce data leakage
3. Training, validation, and test spliting
4. Data set docmentation


### Clasification Experiments (M2-M3)
1. CNN trained from scratch. 
2. Frozen pre-trained backbone. 
3. Full fine-tune of a pre-trained model. 
4. Evaluation of pre -trained image classification models. 
5. Experiment tracling using Weights and Biases (W&B)

### Evaluation Results
Experiment results are available in the results /directory

## Future Work - PHase 2
Phase 2 will foces on object detection, image, segmentation, and the development of the web portal for Real/Fake image classification.


## Project Resources

### Dataset Card
[View Dataset Card](docummentation/dataset_card.md)


## Project Structure

```text
Deep-Learning-Based-Detection-of-Fake-Images-in-Saudi-Arabia/
├── notebooks/         # Training and evaluation notebooks
├── results/           # Experiment results and metrics
├── docummentation/    # Dataset card and documentation
├── data/              # Dataset information
├── requirements.txt   # Python dependencies
├── .gitignore         # Git exclusion rules
└── README.md          # Project overview
```


## Installation & Usage

### 1. Requirements

- Python 3
- Jupyter Notebook or Kaggle
- Required Python libraries listed in `requirements.txt`
- Access to the project dataset

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation

The dataset is based on the Renyang/DANI dataset.

Dataset source:
https://huggingface.co/datasets/Renyang/DANI

The notebook expects the prepared dataset to be available in the configured data directory.

Users must update dataset paths according to their execution environment.

### 4. Run the Notebook

1. Open the `notebooks/` directory.
2. Open the Phase 1 implementation notebook.
3. Configure the dataset and output paths.
4. Install the required dependencies.
5. Execute the notebook cells in their intended order.

### 5. Experiments

The Phase 1 implementation includes:

- Pretrained model validation comparisons
- CNN trained from scratch
- Frozen pretrained backbone
- Full fine-tuning
- Model evaluation and error analysis
- Experiment tracking using Weights & Biases (W&B)

### 6. Results

Saved experiment metrics and evaluation results are available in the `results/` directory.

Dataset documentation is available in:

[Dataset Card](docummentation/dataset_card.md)

### 7. Reproducibility

The project uses a fixed random seed of `42`.

Related images are grouped before data splitting to reduce potential data leakage.

The same experimental data splits are used for model comparisons.

The evaluation is exploratory, and the documented dataset limitations should be considered when interpreting results.


## Pretrained Models Comparison

**Evaluation:** Validation | **Images:** 156 | **Resolution:** 224×224 | **Threshold:** 0.5

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Swin_AI_Image | **56.41%** | **56.38%** |
| SigLIP2_Deepfake | 44.23% | 44.21% |
| ViT_Dima | 45.51% | 41.03% |
| SigLIP2_ThreeClass | 39.74% | 37.74% |
| ViT_ThreeClass | 50.64% | 35.80% |

**Best pretrained model:** Swin_AI_Image, selected using validation Macro-F1.


## Training Strategies Comparison

**Evaluation:** Exploratory Test | **Images:** 164 | **Resolution:** 224×224 | **Threshold:** 0.5

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| Pretrained | 59.15% | 59.07% | 60.26% |
| Scratch CNN | 49.39% | 40.63% | 44.97% |
| Frozen Backbone | 89.63% | 89.60% | 96.28% |
| Full Fine-tuning | **97.56%** | **97.56%** | **99.76%** |

**Best model:** Full Fine-tuning, selected based on validation performance.

## Weights & Biases (W&B)

All Phase 1 experiment results, model comparisons, training metrics, and performance visualizations are available on Weights & Biases.

**[View Our W&B Project and Experiment Results](https://forge.coreweave.com/wandb/ifbf-kaust/phase1-real-fake/overview)**

## Team Members

- Fahad Falah H. AlQahtani
- Anoud Mohammd A. Alenzei
- Abdulaziz Salim AlQahtani

**Program:** PGDip Artificial Intelligence 2026  
**Institution:** King Abdullah University of Science and Technology (KAUST)

