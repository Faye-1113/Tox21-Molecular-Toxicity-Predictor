import os, threading
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt
plt.set_loglevel('error')
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from rdkit import RDLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LambdaCallback

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Tox21_Predictor(tk.Tk):
    # ----------------------------- Initialization -----------------------------
    def __init__(self):
        super().__init__()

        # Constants
        self.ALL_TARGETS = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
            "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ]
        self.FP_SIZE = 1024            # Fingerprint bit length
        self.RADIUS = 2                # Morgan fingerprint radius

        # RDKit availability
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            self.rdkit_Chem = Chem
            self.rdkit_AllChem = AllChem
            self.rdkit_available = True
        except ImportError:
            self.rdkit_available = False

        # GUI colors and fonts
        self.BG = "#F0F0F0"
        self.SURFACE = "#FFF"
        self.ACCENT = "#4A6FA5"
        self.ACCENT2 = "#6B8DB5"
        self.TEXT = "#222"
        self.TEXT_DIM = "#666"
        self.SUCCESS = "#5A9E6E"
        self.DANGER = "#C75C5C"
        self.BORDER = "#DDD"
        self.FONT_TITLE = ("Segoe UI", 15, "bold")
        self.FONT_HEAD = ("Segoe UI", 11, "bold")
        self.FONT_BODY = ("Segoe UI", 10)
        self.FONT_SMALL = ("Segoe UI", 9)
        self.FONT_MONO = ("Courier New", 10)
        self.PALETTE = {"pos": "#E05252", "neg": "#4DA8DA", "grid": "#EEE", "text": "#333", "bg": "#FAFAFA"}

        # Data related attributes
        self.model = None               # trained model for single target
        self.X_test = None
        self.y_test = None
        self.y_prob = None
        self.cm_vals = None
        self.last_target = ""
        self.best_threshold = 0.65
        self.avg_metrics = None
        self.all_models = []            # list of models for all 12 targets
        self.all_y_tests = []
        self.all_y_probs = []
        self.all_target_names = []
        self.df_raw = None              # raw DataFrame from CSV
        self.fp_cache = {}              # SMILES -> fingerprint cache
        self.failed_count = 0

        # GUI variables
        self.title("Tox21 Molecular Toxicity Predictor")
        self.csv_path = tk.StringVar(value="tox21.csv")
        self.target_var = tk.StringVar(value="NR-AR")

        # Build the GUI
        self.build_ui()

    # ----------------------------- Helper Methods -----------------------------
    def get_fingerprint(self, smiles):
        # Convert SMILES to Morgan fingerprint; returns None if RDKit unavailable.
        if not smiles or not isinstance(smiles, str) or not smiles.strip():
            return None
        if not self.rdkit_available:
            print("Error: RDKit is not available. Cannot generate fingerprint.")
            return None
        try:
            mol = self.rdkit_Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = self.rdkit_AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.RADIUS, nBits=self.FP_SIZE)
            return np.array(fp, dtype=np.float32)
        except Exception as e:
            print(f"Error generating fingerprint for SMILES {smiles}: {e}")
            return None

    def compute_metrics(self, y_true, y_pred):
        # Compute TP, FP, FN, TN and derived metrics from confusion matrix.
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = (tp+tn)/(tp+tn+fp+fn+1e-12)
        prec = tp/(tp+fp+1e-12)
        rec = tp/(tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        spec = tn/(tn+fp+1e-12)
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "Accuracy": acc, "Precision": prec, "Recall": rec,
                "F1-Score": f1, "Specificity": spec}

    # ----------------------------- Data Preprocessing -----------------------------
    def load_dataset(self, progress_cb=None):
        # Read CSV, generate fingerprints for unique SMILES, cache them.
        df = pd.read_csv(self.csv_path.get())
        self.df_raw = df
        if "smiles" not in df.columns:
            raise ValueError("CSV file must contain 'smiles' column.")
        unique_smiles = df["smiles"].dropna().unique()
        total = len(unique_smiles)
        self.fp_cache = {}
        self.failed_count = 0
        for i, smi in enumerate(unique_smiles):
            fp = self.get_fingerprint(smi)
            self.fp_cache[smi] = fp
            if fp is None:
                self.failed_count += 1
            if progress_cb:
                progress_cb(i+1, total)
        return self

    def get_available_targets(self):
        # Return list of targets that exist in the loaded DataFrame.
        if self.df_raw is None:
            raise RuntimeError("Call load_dataset first.")
        result = []
        for t in self.ALL_TARGETS:
            if t in self.df_raw.columns:
                result.append(t)
        return result

    def get_split(self, target, train_ratio=0.7, val_ratio=0.15, random_seed=42):
        # Split data into train/validation/test sets for a given target.
        from sklearn.model_selection import train_test_split
        if self.df_raw is None:
            raise RuntimeError("Call load_dataset first.")
        if target not in self.df_raw.columns:
            raise ValueError(f"Target '{target}' not found in dataset.")
        df = self.df_raw.dropna(subset=[target]).copy()
        df[target] = df[target].astype(int)
        X_list, y_list = [], []
        for _, row in df.iterrows():
            smi = row["smiles"]
            fp = self.fp_cache.get(smi)
            if fp is not None:
                X_list.append(fp)
                y_list.append(int(row[target]))
        if not X_list:
            raise ValueError("No valid fingerprints found for this target.")
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=1-train_ratio-val_ratio, random_state=random_seed, stratify=y)
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_seed, stratify=y_train_val)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def dataset_stats(self, target):
        # Return number of positive/negative samples for a target.
        if self.df_raw is None:
            raise RuntimeError("Call load_dataset first.")
        col = self.df_raw[target].dropna().astype(int)
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        return {"total_labelled": len(col), "positive": n_pos,
                "negative": n_neg, "pos_ratio": n_pos/len(col) if len(col) else 0}

    # ----------------------------- Model Building and Training -----------------------------
    def build_model(self, input_dim, learning_rate, lambda_reg):
        # Construct a Keras MLP model with batch norm, dropout, L2 regularization.
        model = Sequential(name="Tox21_MLP")
        model.add(Input(shape=(input_dim,)))  # Input layer
        model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.l2(lambda_reg)))  # First hidden layer
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, kernel_regularizer=tf.keras.regularizers.l2(lambda_reg)))  # Second hidden layer
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))  # Third hidden layer
        model.add(Dense(1, activation='sigmoid'))  # Output layer
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
 
    def train_model(self, X, y, X_val=None, y_val=None, status_callback=None,
                    epochs=100, batch_size=128, l2=2e-4, lr=0.0001):
        # Train the model with early stopping, class weights, and optional GUI callback.
        n, d = X.shape
        model = self.build_model(d, learning_rate=lr, lambda_reg=l2)
        # Set validation data only if both X_val and y_val are provided
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
 
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y.ravel())
            class_weight_dict = dict(zip(np.unique(y), class_weights))
        except Exception:
            class_weight_dict = None

        # Construct callbacks: early stopping only if validation data exists
        if validation_data is not None:
            callbacks = [early_stop]
        else:
            callbacks = []
       
        if status_callback:
            epoch_callback = LambdaCallback(
                on_epoch_end=lambda epoch, logs: status_callback(
                    epoch+1, logs.get('loss', 0), logs.get('accuracy', 0),
                    logs.get('val_loss', None), logs.get('val_accuracy', None)
                )
            )
            callbacks.append(epoch_callback)

        history = model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
            shuffle=True
        )

        model.loss_history = history.history.get('loss', [])
        model.val_loss_history = history.history.get('val_loss', [])
        model.train_acc_history = history.history.get('accuracy', [])
        model.val_acc_history = history.history.get('val_accuracy', [])

        self.loss_history = model.loss_history
        self.val_loss_history = model.val_loss_history
        self.train_acc_history = model.train_acc_history
        self.val_acc_history = model.val_acc_history

        return model

    # ----------------------------- Threshold Selection -----------------------------
    def best_th(self, y_true, y_prob):
        # Select threshold that maximizes F1 score on validation set.
        ts = np.sort(np.unique(y_prob))[::-1]
        best_f1 = 0
        best = 0.65
        for t in ts:
            yp = (y_prob >= t).astype(int)
            tp = np.sum((yp == 1) & (y_true == 1))
            fp = np.sum((yp == 1) & (y_true == 0))
            fn = np.sum((yp == 0) & (y_true == 1))
            if tp + fp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best = t
        return best

    # ----------------------------- Plotting Methods -----------------------------
    def base_fig(self, w=6, h=4.5):
        # Create a matplotlib figure with consistent style.
        fig, ax = plt.subplots(figsize=(w, h), facecolor=self.PALETTE['bg'])
        ax.set_facecolor(self.PALETTE['bg'])
        ax.tick_params(colors=self.PALETTE['text'])
        ax.yaxis.grid(True, color=self.PALETTE['grid'], zorder=0)
        ax.set_axisbelow(True)
        return fig, ax

    def plot_confusion_matrix(self, cm_array, target):
        # Draw a custom confusion matrix with colored blocks.
        labels = [["TN","FP"],["FN","TP"]]
        colours = [["#4DA8DA","#F4A56A"],["#F4A56A","#E05252"]]
        fig,ax = plt.subplots(figsize=(5,4), facecolor=self.PALETTE['bg'])
        ax.set_facecolor(self.PALETTE['bg'])
        for i in range(2):
            for j in range(2):
                ax.add_patch(mpatches.FancyBboxPatch((j+0.05,1-i+0.05),0.9,0.9,
                            boxstyle="round,pad=0.02", facecolor=colours[i][j],
                            edgecolor='white',linewidth=2))
                ax.text(j+0.5,1.5-i,f"{labels[i][j]}\n{cm_array[i,j]}",ha='center',va='center',
                        fontsize=14,fontweight='bold',color='white')
        ax.set_xlim(0,2); ax.set_ylim(0,2)
        ax.set_xticks([0.5,1.5])
        ax.set_xticklabels(["Pred: 0","Pred: 1"],fontsize=11)
        ax.set_yticks([0.5,1.5])
        ax.set_yticklabels(["Actual: 1","Actual: 0"],fontsize=11)
        ax.set_title(f"Confusion Matrix — {target}",fontsize=13,color=self.PALETTE['text'])
        for sp in ax.spines.values(): 
            sp.set_visible(False)
        fig.tight_layout()
        return fig

    def plot_loss_curve(self, loss_history, target, val_loss_history=None):
        # Plot training loss and validation loss.
        fig, ax = self.base_fig(6, 4)
        ep = range(1, len(loss_history) + 1)
        ax.plot(ep, loss_history, color="#4DA8DA", linewidth=2, label="Training Loss")
        # Check if validation loss contains any non-None values
        if val_loss_history is not None and isinstance(val_loss_history, list):
            has_valid = False
            for v in val_loss_history:
                if v is not None:
                    has_valid = True
                    break
            if has_valid:
                clean_ep = []
                clean_val = []
                for i, v in enumerate(val_loss_history, start=1):
                    if v is not None:
                        clean_ep.append(i)
                        clean_val.append(v)
                if clean_val:
                    ax.plot(clean_ep, clean_val, color="#E05252", linewidth=2,
                            linestyle="--", label="Validation Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"Loss Curve — {target}", fontsize=13)
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_acc_curve(self, train_acc_history, val_acc_history, target):
        # Plot training accuracy and optional validation accuracy.
        fig, ax = self.base_fig(6, 4)
        epochs = range(1, len(train_acc_history) + 1)
        ax.plot(epochs, train_acc_history, color="#4DA8DA", linewidth=2, label="Training Accuracy")
        if val_acc_history is not None and isinstance(val_acc_history, list):
            has_valid = False
            for v in val_acc_history:
                if v is not None:
                    has_valid = True
                    break
            if has_valid:
                clean_ep = []
                clean_val = []
                for i, v in enumerate(val_acc_history, start=1):
                    if v is not None:
                        clean_ep.append(i)
                        clean_val.append(v)
                if clean_val:
                    ax.plot(clean_ep, clean_val, color="#E05252", linewidth=2,
                            linestyle="--", label="Validation Accuracy")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy Curve — {target}", fontsize=13)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)
        fig.tight_layout()
        return fig

    def plot_roc_curve(self, y_true, y_prob, target):
        # Plot ROC curve and display AUC.
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig,ax = self.base_fig(5.5,4.5)
        ax.plot(fpr, tpr, color='#E05252', linewidth=2, label=f'AUC={roc_auc:.3f}')
        ax.plot([0,1],[0,1],'--',color='#AAAAAA')
        ax.fill_between(fpr, tpr, alpha=0.12, color='#E05252')
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC Curve — {target}", fontsize=13)
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_loss_curves_comparison(self, models, names):
        # Overlay training/validation loss curves for multiple models.
        fig, ax = self.base_fig(12, 7)
        colors = cm.tab20(np.linspace(0, 1, len(models)))
        for m, n, c in zip(models, names, colors):
            ep = range(1, len(m.loss_history) + 1)
            ax.plot(ep, m.loss_history, color=c, linewidth=1.5, label=n)
            # Plot validation loss if present
            if hasattr(m, 'val_loss_history') and isinstance(m.val_loss_history, list):
                has_valid = False
                for v in m.val_loss_history:
                    if v is not None:
                        has_valid = True
                        break
                if has_valid:
                    clean_ep = []
                    clean_val = []
                    for i, v in enumerate(m.val_loss_history, start=1):
                        if v is not None:
                            clean_ep.append(i)
                            clean_val.append(v)
                    if clean_val:   
                        ax.plot(clean_ep, clean_val, color=c, linewidth=1.0, linestyle="--", alpha=0.4)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Loss Comparison — All Targets (solid=train, dashed=val)")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        return fig

    def plot_acc_curves_comparison(self, models, names):
        # Overlay training/validation accuracy curves for multiple models.
        fig, ax = self.base_fig(12, 7)
        colors = cm.tab20(np.linspace(0, 1, len(models)))
        for m, n, c in zip(models, names, colors):
            ep_train = range(1, len(m.train_acc_history) + 1)
            ax.plot(ep_train, m.train_acc_history, color=c, linewidth=1.5, label=f"{n} (train)")
            if hasattr(m, 'val_acc_history') and isinstance(m.val_acc_history, list):
                has_valid = False
                for v in m.val_acc_history:
                    if v is not None:
                        has_valid = True
                        break
                if has_valid:
                    clean_ep = []
                    clean_val = []
                    for i, v in enumerate(m.val_acc_history, start=1):
                        if v is not None:
                            clean_ep.append(i)
                            clean_val.append(v)
                    if clean_val: 
                        ax.plot(clean_ep, clean_val, color=c, linewidth=1.5, linestyle="--", label=f"{n} (val)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Comparison — All Targets (solid=train, dashed=val)")
        ax.legend(fontsize=6, ncol=3)
        fig.tight_layout()
        return fig

    def plot_roc_curves_comparison(self, y_tests, y_probs, names):
        # Overlay ROC curves for multiple models.
        fig,ax = self.base_fig(8,6)
        colors = cm.tab20(np.linspace(0, 1, len(y_tests)))
        for yt, yp, n, c in zip(y_tests, y_probs, names, colors):
            fpr, tpr, _ = roc_curve(yt, yp)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=c, linewidth=1.8, label=f'{n} (AUC={roc_auc:.3f})')
        ax.plot([0,1],[0,1],'--',color='#AAAAAA')
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Comparison — All Targets")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        return fig

    # ----------------------------- GUI Construction -----------------------------
    def set_all_buttons_state(self, load_state, train_state, train_all_state):
        # Enable/disable main action buttons.
        try:
            self.load_btn.config(state=load_state)
            self.train_btn.config(state=train_state)
            self.train_all_btn.config(state=train_all_state)
        except Exception:
            pass

    def build_ui(self):
        # Create all GUI widgets.
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = tk.Frame(self, bg=self.ACCENT, height=50)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        tk.Label(header, text="Tox21 Molecular Toxicity Predictor", font=self.FONT_TITLE,
                 fg="white", bg=self.ACCENT).pack(side="left", padx=20, pady=10)

        main_container = tk.Frame(self, bg=self.BG)
        main_container.grid(row=1, column=0, sticky="nsew")

        # Left panel (cards)
        left_frame = tk.Frame(main_container, bg=self.BG, width=700, height=900)
        left_frame.pack(side="left", fill="y")
        left_frame.pack_propagate(False)
        self.left_frame = left_frame

        # Right panel (notebook)
        right_frame = tk.Frame(main_container, bg=self.BG)
        right_frame.pack(side="left", fill="both", expand=True)

        # Helper functions for consistent styling
        def sbutton(parent, text, cmd):
            b = tk.Button(parent, text=text, command=cmd, bg=self.ACCENT, fg="white",
                          activebackground=self.ACCENT2, activeforeground="white",
                          font=self.FONT_BODY, relief="flat", cursor="hand2", padx=15, pady=5, bd=0)
            b.bind("<Enter>", lambda e: b.config(bg=self.ACCENT2))
            b.bind("<Leave>", lambda e: b.config(bg=self.ACCENT))
            return b

        def card(parent):
            return tk.Frame(parent, bg=self.SURFACE, bd=1, relief="solid", highlightbackground=self.BORDER)

        # Dataset Card
        c = card(left_frame)
        c.pack(fill="x", pady=(0,10))
        tk.Label(c, text="Dataset", font=self.FONT_HEAD, fg=self.ACCENT, bg=self.SURFACE).pack(anchor="w", padx=12, pady=(8,4))
        r = tk.Frame(c, bg=self.SURFACE)
        r.pack(fill="x", padx=12, pady=(0,8))
        e = tk.Entry(r, textvariable=self.csv_path, bg=self.BG, fg=self.TEXT, insertbackground=self.TEXT,
                     relief="flat", font=self.FONT_SMALL, bd=3)
        e.pack(side="left", fill="x", expand=True)
        tk.Button(r, text="Browse", command=self.browse_csv, bg=self.BORDER, fg=self.TEXT,
                  relief="flat", font=self.FONT_SMALL, padx=6, cursor="hand2").pack(side="left", padx=(4,0))
        self.load_btn = sbutton(c, "Load Dataset", self.start_load)
        self.load_btn.pack(fill="x", padx=12, pady=(0,8))
        self.load_prog = ttk.Progressbar(c, mode="determinate")
        self.load_prog.pack(fill="x", padx=12, pady=(0,8))
        self.load_status = tk.Label(c, text="Not loaded", font=self.FONT_SMALL, fg=self.TEXT_DIM, bg=self.SURFACE)
        self.load_status.pack(anchor="w", padx=12, pady=(0,10))

        # Target Card
        c2 = card(left_frame)
        c2.pack(fill="x", pady=(0,10))
        tk.Label(c2, text="Toxicity Target", font=self.FONT_HEAD, fg=self.ACCENT, bg=self.SURFACE).pack(anchor="w", padx=12, pady=(8,4))
        self.target_cb = ttk.Combobox(c2, textvariable=self.target_var, values=self.ALL_TARGETS,
                                      state="readonly", font=self.FONT_BODY)
        self.target_cb.pack(fill="x", padx=12, pady=(0,10))

        # Training Card
        c3 = card(left_frame)
        c3.pack(fill="x", pady=(0,10))
        tk.Label(c3, text="Training", font=self.FONT_HEAD, fg=self.ACCENT, bg=self.SURFACE).pack(anchor="w", padx=12, pady=(8,4))
        self.train_btn = sbutton(c3, "Train the Target", self.start_train)
        self.train_btn.pack(fill="x", padx=12, pady=(4,4))
        self.train_all_btn = sbutton(c3, "Train ALL 12 Targets", self.start_train_all)
        self.train_all_btn.pack(fill="x", padx=12, pady=(4,4))
        self.train_status = tk.Label(c3, text="Idle", font=self.FONT_SMALL, fg=self.TEXT_DIM, bg=self.SURFACE)
        self.train_status.pack(anchor="w", padx=12, pady=(0,10))

        # Predict Card
        c4 = card(left_frame)
        c4.pack(fill="x")
        tk.Label(c4, text="Predict SMILES", font=self.FONT_HEAD, fg=self.ACCENT, bg=self.SURFACE).pack(anchor="w", padx=12, pady=(8,4))
        self.smiles_entry = tk.Entry(c4, bg=self.BG, fg=self.TEXT, insertbackground=self.TEXT,
                                     relief="flat", font=self.FONT_MONO, bd=3)
        self.smiles_entry.pack(fill="x", padx=12, pady=6)
        self.smiles_entry.insert(0, "CCOc1ccc2nc(S(N)(=O)=O)sc2c1")
        sbutton(c4, "Predict", self.predict_smiles).pack(fill="x", padx=12, pady=(0,12))

        # Notebook (tabs for results)
        nb = ttk.Notebook(right_frame)
        nb.pack(fill="both", expand=True)
        self.tab_metrics = tk.Frame(nb, bg=self.BG); nb.add(self.tab_metrics, text="Metrics")
        self.tab_cm = tk.Frame(nb, bg=self.BG); nb.add(self.tab_cm, text="Confusion Matrix")
        self.tab_loss = tk.Frame(nb, bg=self.BG); nb.add(self.tab_loss, text="Loss Curve")
        self.tab_acc = tk.Frame(nb, bg=self.BG); nb.add(self.tab_acc, text="Accuracy Curve")
        self.tab_roc = tk.Frame(nb, bg=self.BG); nb.add(self.tab_roc, text="ROC Curve")
        self.tab_pred = tk.Frame(nb, bg=self.BG); nb.add(self.tab_pred, text="Prediction")
        self.notebook = nb

    # ----------------------------- File Browsing -----------------------------
    def browse_csv(self):
        # Open file dialog to select dataset CSV.
        p = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv"),("All Files","*.*")])
        if p: 
            self.csv_path.set(p)

    # ----------------------------- Loading Thread -----------------------------
    def start_load(self):
        # Start background thread to load dataset.
        if not os.path.isfile(self.csv_path.get()):
            messagebox.showerror("File Not Found", f"Cannot find {self.csv_path.get()}")
            return
        self.set_all_buttons_state("disabled", "disabled", "disabled")
        self.load_btn.config(text="Loading…")
        self.load_prog["value"] = 0
        threading.Thread(target=self.load_worker, daemon=True).start()

    def load_worker(self):
        # Worker thread: read CSV, compute fingerprints, update GUI via after().
        try:
            df = pd.read_csv(self.csv_path.get())
            self.df_raw = df
            if "smiles" not in df.columns:
                raise ValueError("CSV file must contain 'smiles' column.")
            unique_smiles = df["smiles"].dropna().unique()
            total = len(unique_smiles)
            self.fp_cache = {}
            self.failed_count = 0
            for i, smi in enumerate(unique_smiles):
                fp = self.get_fingerprint(smi)
                self.fp_cache[smi] = fp
                if fp is None:
                    self.failed_count += 1
                self.after(0, lambda d=i+1, t=total: self.load_prog.configure(value=int(d/t*100)))
                self.after(0, lambda d=i+1, t=total: self.load_status.config(text=f"Computing fingerprints… {d}/{t}"))
            if self.failed_count:
                fail_info = f" (fingerprints failed: {self.failed_count})"
            else:
                fail_info = ""
            self.after(0, lambda: self.load_status.config(text=f"Loaded {len(self.get_available_targets())} targets{fail_info}", fg=self.SUCCESS))
        except Exception as ex:
            err_msg = str(ex)
            self.after(0, lambda: self.load_status.config(text=f"Error: {err_msg}", fg=self.DANGER))
        finally:
            self.after(0, lambda: self.load_btn.config(text="Load Dataset"))
            self.after(0, lambda: self.set_all_buttons_state("normal", "normal", "normal"))

    # ----------------------------- Training (Single Target) Thread -----------------------------
    def start_train(self):
        # Start background thread to train a model for the selected target.
        if self.df_raw is None:
            messagebox.showwarning("Dataset Not Loaded", "Please load dataset first!")
            return
        self.set_all_buttons_state("disabled", "disabled", "disabled")
        self.train_btn.config(text="Training…")
        threading.Thread(target=self.train_worker, args=(self.target_var.get(),), daemon=True).start()

    def train_worker(self, target):
        # Worker thread: prepare data, train model, compute metrics, update tabs.
        try:
            self.after(0, lambda: self.train_status.config(text="Preparing data…", fg=self.TEXT_DIM))
            X_tr, X_val, X_te, y_tr, y_val, y_te = self.get_split(target)
            stats = self.dataset_stats(target)

            def status_cb(epoch, loss, acc, val_loss, val_acc):
                msg = f"Epoch {epoch} loss={loss:.4f} acc={acc:.4f}"
                self.after(0, lambda: self.train_status.config(text=msg))

            model = self.train_model(X_tr, y_tr, X_val=X_val, y_val=y_val, status_callback=status_cb)

            if hasattr(model, 'predict'):
                yp_val = model.predict(X_val, verbose=0).flatten()
            else:
                yp_val = model.predict_proba(X_val)
            
            th = self.best_th(y_val, yp_val)
            self.best_threshold = th
            
            if hasattr(model, 'predict'):
                yp_test = model.predict(X_te, verbose=0).flatten()
            else:
                yp_test = model.predict_proba(X_te)
            
            y_pred_test = (yp_test >= th).astype(int)
            # Compute test AUC
            if len(np.unique(y_te)) > 1:
                test_auc = roc_auc_score(y_te, yp_test)
            else:
                test_auc = np.nan
            metrics = self.compute_metrics(y_te, y_pred_test)
            metrics['AUC'] = test_auc

            self.model = model
            self.X_test = X_te
            self.y_test = y_te
            self.y_prob = yp_test
            self.cm_vals = metrics
            self.last_target = target

            self.all_models.clear()
            self.all_y_tests.clear()
            self.all_y_probs.clear()
            self.all_target_names.clear()

            self.after(0, lambda: self.train_status.config(text=f"Best th={th:.2f} Test Acc={metrics['Accuracy']:.3f} F1={metrics['F1-Score']:.3f}", fg=self.SUCCESS))
            self.after(0, lambda: self.update_tabs(metrics, model, y_te, yp_test, target, stats))
        except Exception as ex:
            err_msg = str(ex)
            self.after(0, lambda: self.train_status.config(text=f"Error: {err_msg}", fg=self.DANGER))
        finally:
            self.after(0, lambda: self.train_btn.config(text="Train the Target"))
            self.after(0, lambda: self.set_all_buttons_state("normal", "normal", "normal"))

    # ----------------------------- Training (All Targets) Thread -----------------------------
    def start_train_all(self):
        # Start background thread to train models for all 12 targets.
        if self.df_raw is None:
            messagebox.showwarning("Warning", "Please load dataset first!")
            return
        self.set_all_buttons_state("disabled", "disabled", "disabled")
        self.train_all_btn.config(text="Training ALL...")
        threading.Thread(target=self.train_all_worker, daemon=True).start()

    def train_all_worker(self):
        # Worker thread: sequentially train each target, collect metrics, show average.
        try:
            targs = self.ALL_TARGETS
            self.all_models.clear()
            self.all_y_tests.clear()
            self.all_y_probs.clear()
            self.all_target_names = targs
            self.model = None
            metrics_list = []
            total = len(targs)

            for i, tg in enumerate(targs):
                self.after(0, lambda i=i, tg=tg: self.train_status.config(text=f"Training {i+1}/{total} | {tg}"))
                X_tr, X_val, X_te, y_tr, y_val, y_te = self.get_split(tg)

                # Capture current i and tg using default arguments to avoid closure late binding
                def status_cb(epoch, loss, acc, val_loss, val_acc, idx=i, name=tg):
                    msg = f"[{idx+1}/{total}] {name} Epoch {epoch} loss={loss:.4f} acc={acc:.4f}"
                    self.after(0, lambda: self.train_status.config(text=msg))

                model = self.train_model(X_tr, y_tr, X_val=X_val, y_val=y_val, status_callback=status_cb)

                if hasattr(model, 'predict'):
                    yp_val = model.predict(X_val, verbose=0).flatten()
                else:
                    yp_val = model.predict_proba(X_val)
                
                th = self.best_th(y_val, yp_val)
                model.best_threshold = th
                
                if hasattr(model, 'predict'):
                    yp_test = model.predict(X_te, verbose=0).flatten()
                else:
                    yp_test = model.predict_proba(X_te)
                
                y_pred_test = (yp_test >= th).astype(int)
                if len(np.unique(y_te)) > 1:
                    test_auc = roc_auc_score(y_te, yp_test)
                else:
                    test_auc = np.nan
                metrics = self.compute_metrics(y_te, y_pred_test)
                metrics['AUC'] = test_auc
                metrics_list.append(metrics)
                self.all_models.append(model)
                self.all_y_tests.append(y_te)
                self.all_y_probs.append(yp_test)

            if len(metrics_list) == total:
                avg = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0, "Specificity": 0, "AUC": 0}
                for d in metrics_list:
                    for k in avg:
                        if not np.isnan(d[k]):
                            avg[k] += d[k]
                for k in avg:
                    avg[k] = round(avg[k] / len(metrics_list), 4) if len(metrics_list) > 0 else np.nan
                self.avg_metrics = avg
                self.after(0, self.show_all)
                self.after(0, lambda: self.train_status.config(text=f"All 12 done! Avg F1={avg['F1-Score']:.3f}", fg=self.SUCCESS))
            else:
                self.after(0, lambda: self.train_status.config(text="Training incomplete (some targets missing).", fg=self.DANGER))
        except Exception as ex:
            err_msg = str(ex)
            self.after(0, lambda: messagebox.showerror("Error", err_msg))
        finally:
            self.after(0, lambda: self.train_all_btn.config(text="Train ALL 12 Targets"))
            self.after(0, lambda: self.set_all_buttons_state("normal", "normal", "normal"))

    # ----------------------------- GUI Update Helpers -----------------------------
    def update_tabs(self, metrics, model, y_test, y_prob, target, stats):
        # Update Metrics, Confusion Matrix, Loss, Accuracy, ROC tabs after training.
        try:
            # Clear previous content in Metrics tab
            for w in self.tab_metrics.winfo_children():
                w.destroy()
            # Performance metrics
            mf = tk.Frame(self.tab_metrics, bg=self.SURFACE, bd=1, relief="solid", highlightbackground=self.BORDER)
            mf.pack(fill="x", padx=16, pady=(0,12))
            tk.Label(mf, text=f"Performance Metrics — {target}", font=self.FONT_HEAD, fg=self.ACCENT, bg=self.SURFACE).pack(anchor="center", padx=12, pady=(10,6))
            metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "AUC"]
            for k in metric_keys:
                r = tk.Frame(mf, bg=self.SURFACE)
                r.pack(fill="x", padx=12, pady=3)
                tk.Label(r, text=f"{k}:", font=self.FONT_BODY, fg=self.TEXT_DIM, bg=self.SURFACE).pack(side="left")
                value = metrics.get(k, np.nan)
                if np.isnan(value):
                    display_value = "N/A"
                else:
                    display_value = f"{value:.4f}"
                tk.Label(r, text=display_value, font=self.FONT_BODY, fg=self.ACCENT, bg=self.SURFACE).pack(side="right")
            # Embed plots
            cm_array = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
            self.embed(self.tab_cm, self.plot_confusion_matrix(cm_array, target))
            self.embed(self.tab_loss, self.plot_loss_curve(self.loss_history, target, self.val_loss_history))
            self.embed(self.tab_acc, self.plot_acc_curve(self.train_acc_history, self.val_acc_history, target))
            self.embed(self.tab_roc, self.plot_roc_curve(y_test, y_prob, target))
        except Exception as e:
            print("Tab update error:", e)

    def embed(self, tab, fig):
        # Embed a matplotlib figure into a Tkinter tab.
        try:
            for w in tab.winfo_children():
                w.destroy()
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        except Exception as e:
            print("Plot embed error:", e)

    def show_all(self):
        # Show average performance page and comparison plots after training all targets.
        try:
            self.show_avg_page()
            self.embed(self.tab_loss, self.plot_loss_curves_comparison(self.all_models, self.all_target_names))
            self.embed(self.tab_acc, self.plot_acc_curves_comparison(self.all_models, self.all_target_names))
            self.embed(self.tab_roc, self.plot_roc_curves_comparison(self.all_y_tests, self.all_y_probs, self.all_target_names))
        except Exception as e:
            print("Show all error:", e)

    def show_avg_page(self):
        # Display average metrics table for 12 targets.
        try:
            for w in self.tab_metrics.winfo_children():
                w.destroy()
            f = tk.Frame(self.tab_metrics, bg=self.SURFACE, bd=1, relief="solid", highlightbackground=self.BORDER)
            f.pack(fill="x", padx=40, pady=40)
            tk.Label(f, text="Average Performance (12 Tox21 Targets)", font=self.FONT_HEAD, fg=self.ACCENT, bg=self.SURFACE).pack(pady=(20,10))
            metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "AUC"]
            for k in metric_keys:
                r = tk.Frame(f, bg=self.SURFACE)
                r.pack(fill="x", padx=30, pady=6)
                tk.Label(r, text=f"{k}:", font=self.FONT_BODY, fg=self.TEXT_DIM, bg=self.SURFACE).pack(side="left")
                value = self.avg_metrics.get(k, np.nan)
                display_value = f"{value:.4f}" if not np.isnan(value) else "N/A"
                tk.Label(r, text=display_value, font=self.FONT_BODY, fg=self.SUCCESS, bg=self.SURFACE).pack(side="right")
            self.notebook.select(0)
        except Exception as e:
            print("Show avg page error:", e)

    # ----------------------------- Prediction Methods -----------------------------
    def predict_smiles(self):
        # Start background thread to predict toxicity for input SMILES.
        smi = self.smiles_entry.get().strip()
        if not smi:
            messagebox.showwarning("Empty Input", "Please enter a SMILES string.")
            return
        # Clear previous content in Prediction tab
        for w in self.tab_pred.winfo_children():
            w.destroy()
        canvas = tk.Canvas(self.tab_pred, bg=self.BG, highlightthickness=0)
        vbar = tk.Scrollbar(self.tab_pred, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        inner_frame = tk.Frame(canvas, bg=self.BG)
        self.pred_window = canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(self.pred_window, width=e.width))
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        self.pred_canvas = canvas
        self.pred_inner = inner_frame
        tk.Label(inner_frame, text="Processing...", font=("Segoe UI",12), bg=self.BG, fg=self.TEXT_DIM).pack(pady=20)

        if len(self.all_models) == 12:
            threading.Thread(target=self.predict_all_worker, args=(smi, inner_frame), daemon=True).start()
        elif self.model is not None:
            threading.Thread(target=self.predict_single_worker, args=(smi, inner_frame), daemon=True).start()
        else:
            messagebox.showwarning("No Model", "Please train a model first.")

    def predict_single_worker(self, smi, parent_frame):
        # Worker for single target prediction.
        try:
            fp = self.get_fingerprint(smi)
            if fp is None:
                self.after(0, lambda: self.show_prediction_error(parent_frame, "Invalid SMILES"))
                return
            if hasattr(self.model, 'predict'):
                prob = self.model.predict(fp.reshape(1, -1), verbose=0)[0][0]
            else:
                prob = self.model.predict_proba(fp.reshape(1, -1))[0][0]
            th = getattr(self.model, 'best_threshold', self.best_threshold)
            pred = int(prob >= th)
            col = self.DANGER if pred else self.SUCCESS
            res_text = "TOXIC" if pred else "NON-TOXIC"
            detail = f"Target: {self.last_target}\nProbability: {prob:.4f}\nBest threshold: {th:.2f}"
            self.after(0, lambda: self.show_prediction_result(parent_frame, res_text, detail, col))
        except Exception as ex:
            err_msg = str(ex)
            self.after(0, lambda: self.show_prediction_error(parent_frame, err_msg))
        self.notebook.select(5)

    def predict_all_worker(self, smi, parent_frame):
        # Worker for multi-target prediction (all 12 models).
        try:
            fp = self.get_fingerprint(smi)
            if fp is None:
                self.after(0, lambda: self.show_prediction_error(parent_frame, "Invalid SMILES"))
                return
            X = fp.reshape(1, -1)
            results = []
            has_toxic = False
            for name, model in zip(self.all_target_names, self.all_models):
                if hasattr(model, 'predict'):
                   prob = model.predict(X, verbose=0)[0][0]
                else:
                    prob = model.predict_proba(X)[0][0]
                th = model.best_threshold
                verdict = "TOXIC" if prob >= th else "NON-TOXIC"
                if verdict == "TOXIC":
                    has_toxic = True
                results.append((name, verdict, prob, th))
            col = self.DANGER if has_toxic else self.SUCCESS
            self.after(0, lambda: self.show_prediction_all(parent_frame, results, col))
        except Exception as ex:
            err_msg = str(ex)
            self.after(0, lambda: self.show_prediction_error(parent_frame, err_msg))
        self.notebook.select(5)

    # ----------------------------- Display Prediction Results -----------------------------
    def show_prediction_result(self, parent, title_text, detail_text, color):
        # Display single-target prediction result.
        try:
            for w in parent.winfo_children():
                w.destroy()
            box = tk.Frame(parent, bg=self.SURFACE, bd=1, relief="solid", highlightbackground=self.BORDER)
            box.pack(fill="both", expand=True, padx=20, pady=20)
            tk.Label(box, text="Prediction Result", font=self.FONT_HEAD, bg=self.SURFACE, fg=self.TEXT_DIM).pack(pady=(16,4))
            tk.Frame(box, bg=self.BORDER, height=1).pack(fill="x", padx=20)
            tk.Label(box, text=title_text, font=("Segoe UI", 24, "bold"), bg=self.SURFACE, fg=color).pack(pady=(20,8))
            tk.Label(box, text=detail_text, font=self.FONT_BODY, bg=self.SURFACE, fg=self.TEXT).pack(pady=(0,8))
            tk.Frame(box, bg=self.BORDER, height=1).pack(fill="x", padx=20)
            tk.Label(box, text=f"SMILES: {self.smiles_entry.get().strip()}", font=self.FONT_MONO, bg=self.SURFACE, fg=self.TEXT_DIM,
                    wraplength=600, justify="center").pack(pady=(10,16))
        except Exception as e:
            print("Show prediction error:", e)

    def show_prediction_all(self, parent, results, color):
        # Display multi-target prediction result table.
        try:
            for w in parent.winfo_children():
                w.destroy()
            box = tk.Frame(parent, bg=self.SURFACE, bd=1, relief="solid", highlightbackground=self.BORDER)
            box.pack(fill="both", expand=True, padx=20, pady=20)
            tk.Label(box, text="Multi-Target Prediction Results", font=self.FONT_HEAD, bg=self.SURFACE, fg=self.ACCENT).pack(pady=10)
            tk.Frame(box, bg=self.BORDER, height=1).pack(fill="x", padx=20)
            header = tk.Frame(box, bg=self.SURFACE)
            header.pack(fill="x", padx=20, pady=5)
            tk.Label(header, text="Target", width=18, anchor="w", font=self.FONT_BODY, bg=self.SURFACE, fg=self.TEXT_DIM).pack(side="left")
            tk.Label(header, text="Toxicity", width=12, anchor="w", font=self.FONT_BODY, bg=self.SURFACE, fg=self.TEXT_DIM).pack(side="left")
            tk.Label(header, text="Probability", width=12, anchor="w", font=self.FONT_BODY, bg=self.SURFACE, fg=self.TEXT_DIM).pack(side="left")
            tk.Label(header, text="Threshold", width=12, anchor="w", font=self.FONT_BODY, bg=self.SURFACE, fg=self.TEXT_DIM).pack(side="left")
            tk.Frame(box, bg=self.BORDER, height=1).pack(fill="x", padx=20)
            for name, verdict, prob, th in results:
                row = tk.Frame(box, bg=self.SURFACE)
                row.pack(fill="x", padx=20, pady=2)
                tk.Label(row, text=name, width=18, anchor="w", font=self.FONT_BODY, bg=self.SURFACE, fg=self.TEXT).pack(side="left")
                tk.Label(row, text=verdict, width=12, anchor="w", font=self.FONT_BODY, bg=self.SURFACE,
                         fg=self.DANGER if verdict=="TOXIC" else self.SUCCESS).pack(side="left")
                tk.Label(row, text=f"{prob:.4f}", width=12, anchor="w", font=self.FONT_BODY, bg=self.SURFACE, fg=self.TEXT).pack(side="left")
                tk.Label(row, text=f"{th:.2f}", width=12, anchor="w", font=self.FONT_BODY, bg=self.SURFACE, fg=self.TEXT).pack(side="left")
            tk.Frame(box, bg=self.BORDER, height=1).pack(fill="x", padx=20, pady=5)
            tk.Label(box, text=f"SMILES: {self.smiles_entry.get().strip()}", font=self.FONT_MONO, bg=self.SURFACE, fg=self.TEXT_DIM,
                     wraplength=600, justify="center").pack(pady=(10,16))
        except Exception as e:
            print("Show all predictions error:", e)

    def show_prediction_error(self, parent, err_msg):
        # Display prediction error message.
        try:
            for w in parent.winfo_children():
                w.destroy()
            box = tk.Frame(parent, bg=self.SURFACE, bd=1, relief="solid", highlightbackground=self.BORDER)
            box.pack(fill="both", expand=True, padx=20, pady=20)
            tk.Label(box, text="Prediction Error", font=self.FONT_HEAD, bg=self.SURFACE, fg=self.DANGER).pack(pady=10)
            tk.Label(box, text=err_msg, font=self.FONT_BODY, bg=self.SURFACE, fg=self.TEXT).pack(pady=10)
        except Exception as e:
            print("Show error error:", e)


if __name__ == "__main__":
    Predictor1 = Tox21_Predictor()
    Predictor1.mainloop()