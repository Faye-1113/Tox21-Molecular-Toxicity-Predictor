Tox21 Molecular Toxicity Predictor
===================================

Group3  
Group members: Chen Zhifei (1305977), Liu Yichun (1305932)  
Section: CPS\*3320\*W01  


DESCRIPTION
-----------
This project predicts the toxicity of molecules using the Tox21 dataset. 
It classifies each molecule as toxic (1) or non-toxic (0) for 12 biological toxicity targets.

The model uses a Multi-Layer Perceptron (MLP) built with TensorFlow/Keras. 
Molecular structure is converted to fixed-length binary fingerprint vectors using RDKit Morgan fingerprints (if rdkit is installed — recommended)


FILE STRUCTURE
--------------
W01_3 - Tox21 Molecular Toxicity Predictor.zip/  
│  
├── Tox21_Predictor.py          ← Entry point (run this file)  
├── tox21.csv            ← Dataset (must be placed here)   
└── README.txt           ← This file  


REQUIREMENTS  
------------  
- Python 3.7 or higher  
- Required packages (install via pip):  
    pip install numpy pandas matplotlib scikit-learn tensorflow  
- Optional (strongly recommended for molecular fingerprints):  
    pip install rdkit  
   
If RDKit is not installed, fingerprint generation will fail (the program will print an error and skip those molecules).  


HOW TO RUN (Spyder)  
-------------------  
1. Open Spyder IDE  
2. Open Tox21_Predictor.py in Spyder  
3. Click "Browse a working directory" button and set the working directory to the project folder (where tox21.csv is located)  
4. Click the Run button (green triangle) or press F5  
5. In the GUI:  

   a. Click "Load Dataset" to load and process tox21.csv.  
      Wait for the progress bar to complete (fingerprint computation may take some time).  

   b. Select a toxicity target from the dropdown (e.g. NR-AR)  

   c. Click "Train the Target" to train a single target.  
      Training progress is shown in the status text.  

   d. After training, explore the result tabs:  
      • Metrics —  Accuracy/Precision/Recall/F1-Score/Specificity/AUC  
      • Confusion Matrix — coloured matrix visualization  
      • Loss Curve — training loss + validation loss over epochs  
      • Accuracy Curve — training accuracy + validation accuracy  
      • ROC Curve — with AUC score  

   e. To train all 12 targets at once, click "Train ALL 12 Targets".  
      This will take some time — please wait for the status text to show completion.  
      After training, average metrics across all targets are displayed.  

   f. To predict a new molecule, paste a SMILES string in the "Predict SMILES" box and click "Predict".  
      Example SMILES: CCOc1ccc2nc(S(N)(=O)=O)sc2c1  


TOXICITY TARGETS (12)  
---------------------  
NR-AR         Nuclear Receptor — Androgen Receptor  
NR-AR-LBD     NR — Androgen Receptor Ligand Binding Domain  
NR-AhR        NR — Aryl Hydrocarbon Receptor  
NR-Aromatase  NR — Aromatase  
NR-ER         NR — Estrogen Receptor α  
NR-ER-LBD     NR — Estrogen Receptor Ligand Binding Domain  
NR-PPAR-gamma NR — Peroxisome Proliferator-Activated Receptor γ  
SR-ARE        Stress Response — Antioxidant Response Element  
SR-ATAD5      SR — ATAD5  
SR-HSE        SR — Heat Shock Factor Response Element  
SR-MMP        SR — Mitochondrial Membrane Potential  
SR-p53        SR — p53  


MODEL DETAILS  
-------------  
• Architecture: MLP (128 → 64 → 32 → 1) with BatchNormalization and Dropout(0.3)  
• Activation functions: ReLU (hidden layers), Sigmoid (output)  
• Optimizer: Adam  
• Loss: Binary Cross-Entropy + L2 Regularization (λ = 2e-4)  
• Features: 1024-bit molecular fingerprint (Morgan fingerprint / n-gram fallback)  
• Data split: 70% train / 15% validation / 15% test  
• Early stopping: Patience = 15 epochs (monitors validation loss)  
• Class weighting: Balanced (handles imbalanced datasets automatically)  


NOTES  
-----  
• The dataset contains missing labels; rows with NaN for the selected target are automatically excluded before training.  
• Class imbalance (most molecules are non-toxic) is handled by class_weight='balanced'.  
• All metrics (confusion matrix, accuracy, precision, recall, F1, specificity, AUC) are computed using scikit-learn.  
• Training with "Train ALL 12 Targets" takes longer but provides a complete evaluation of the model across all toxicity endpoints.  
• The GUI includes tabs for visualizing results — click any tab after training.  
• Single-target prediction works after training a single target.  
• Multi-target prediction works after training all 12 targets.  


TROUBLESHOOTING  
---------------  
Q: "ModuleNotFoundError: No module named 'rdkit'"  
A: Install rdkit with: pip install rdkit  

Q: "ModuleNotFoundError: No module named 'tensorflow'"  
A: Install tensorflow with: pip install tensorflow  

Q: Dataset not found error  
A: Make sure tox21.csv is in the same folder as Tox21_Prediction.py, and the working directory is set correctly in Spyder.  

Q: Training takes a long time  
A: Normal for 12 targets. Each target trains up to 100 epochs with early stopping. The program will stop early when validation loss stops improving.  

Q: "No model" warning when clicking Predict  
A: Train at least one target (single or all) before making predictions.  
