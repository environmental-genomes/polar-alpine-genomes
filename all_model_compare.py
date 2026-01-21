# -*- coding: utf-8 -*-
"""
Final multi-model evaluation framework
======================================

Models:
- Logistic Regression (LR)
- SVM
- Random Forest (RF)
- MLP (with EarlyStopping & ReduceLROnPlateau)

Outputs:
1) model_metrics_summary_extended.csv
2) {model_name}_classification_report.csv
3) Confusion matrices (raw + normalized) in ONE PDF
4) All Models ROC Comparison (One-vs-Rest) PDF
5) Macro-average ROC comparison PDF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, roc_curve, auc, balanced_accuracy_score, cohen_kappa_score)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ======================================================
# Global configuration
# ======================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

MODELS = ['LR', 'SVM', 'RF', 'MLP']
COLORS = {'LR':'blue','SVM':'green','RF':'purple','MLP':'red'}
# ======================================================
# 1. Data loading
# ======================================================
def load_data(gene_file, group_file):
    df = pd.read_csv(gene_file, index_col=0)
    X = df.values.T
    samples = df.columns.tolist()

    group_df = pd.read_csv(group_file, sep='\t', header=None,
                           names=['sample', 'group'])
    group_df = group_df.set_index('sample').loc[samples]

    le = LabelEncoder()
    y = le.fit_transform(group_df['group'])
    class_names = list(le.classes_)

    X = StandardScaler().fit_transform(X)
    return X, y, class_names

# ======================================================
# 2. MLP model
# ======================================================
def build_mlp(input_dim, num_classes):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ======================================================
# 3. Train & predict
# ======================================================
def train_and_predict(model_name, X_train, y_train, X_test, y_test):
    if model_name == 'LR':
        model = LogisticRegression(max_iter=1000,
                                   class_weight='balanced',
                                   random_state=SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    elif model_name == 'SVM':
        model = SVC(probability=True,
                    class_weight='balanced',
                    random_state=SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    elif model_name == 'RF':
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=SEED
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)



    elif model_name == 'MLP':
        model = build_mlp(X_train.shape[1], len(np.unique(y_train)))

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {i: w for i, w in enumerate(class_weights)}

        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=12,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=4,
                              min_lr=1e-5)
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )

        y_prob = model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=1)

    else:
        raise ValueError("Unknown model")

    return y_pred, y_prob

# ======================================================
# 4. Confusion matrices (raw + normalized)
# ======================================================
def plot_confusion_matrices(y_test, pred_dict, class_names, output_pdf):
    n_models = len(pred_dict)
    fig, axes = plt.subplots(2, n_models,
                             figsize=(4*n_models, 8),
                             sharey=True)

    for col, (model_name, y_pred) in enumerate(pred_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=axes[0, col])
        axes[0, col].set_title(f'{model_name} (Raw)')
        axes[0, col].set_xlabel('Predicted')
        axes[0, col].set_ylabel('True')

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=axes[1, col])
        axes[1, col].set_title(f'{model_name} (Normalized)')
        axes[1, col].set_xlabel('Predicted')
        axes[1, col].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()

# ======================================================
# 5. All Models ROC (One-vs-Rest)
# ======================================================
def plot_all_models_roc(y_test, prob_dict, class_names, output_pdf):
    y_bin = label_binarize(y_test, classes=range(len(class_names)))

    plt.figure(figsize=(9, 7))
    for model_name, y_prob in prob_dict.items():
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr,
                     color=COLORS[model_name],
                     lw=1.5, alpha=0.7,
                     label=f'{model_name}-{cls} (AUC={roc_auc:.2f})')

    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('All Models ROC Comparison (One-vs-Rest)')
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()

# ======================================================
# 6. Macro-average ROC
# ======================================================
def plot_macro_avg_roc(y_test, prob_dict, class_names, output_pdf):
    y_bin = label_binarize(y_test, classes=range(len(class_names)))

    plt.figure(figsize=(8,6))
    for model_name, y_prob in prob_dict.items():
        fpr_list, tpr_list = [], []

        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        all_fpr = np.unique(np.concatenate(fpr_list))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(len(class_names)):
            mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
        mean_tpr /= len(class_names)

        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr,
                 lw=2,
                 color=COLORS[model_name],
                 label=f'{model_name} (Macro-AUC={macro_auc:.3f})')

    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-average ROC Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_pdf)
    plt.close()

# ======================================================
# 7. Main
# ======================================================
def main():
    gene_file = r'\data\PA_transposition.csv'
    group_file = r'\data\group.txt'

    X, y, class_names = load_data(gene_file, group_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=SEED
    )

    metrics = []
    y_pred_dict, y_prob_dict = {}, {}

    for model_name in MODELS:
        print(f'Training {model_name}...')
        y_pred, y_prob = train_and_predict(
            model_name, X_train, y_train, X_test, y_test
        )

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')

        # 在表格增加Balanced Accuracy 与 Cohen’s Kappa
        # 正好补齐当前评估体系中 “类别不平衡稳健性”和“超越随机一致性的判别能力
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        # metrics.append([model_name, acc, macro_f1, micro_f1])
        metrics.append([model_name, acc, macro_f1, micro_f1, balanced_acc, kappa])
        y_pred_dict[model_name] = y_pred
        y_prob_dict[model_name] = y_prob

        # classification report
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True
        )
        pd.DataFrame(report).T.to_csv(
            f'{model_name}_classification_report.csv'
        )

    # metrics summary
    metrics_df = pd.DataFrame(
        metrics,
        # columns=['Model', 'Accuracy', 'Macro-F1', 'Micro-F1']
        columns = ['Model', 'Accuracy', 'Macro-F1', 'Micro-F1', 'Balanced-Accuracy', 'Cohen-Kappa']
    )
    metrics_df.to_csv('all-model_metrics_summary_extended.csv', index=False)

    # plots
    plot_confusion_matrices(
        y_test, y_pred_dict, class_names,
        'all_models_confusion_matrix_raw_and_normalized.pdf'
    )

    plot_all_models_roc(
        y_test, y_prob_dict, class_names,
        'all_models_ROC_one_vs_rest.pdf'
    )

    plot_macro_avg_roc(
        y_test, y_prob_dict, class_names,
        'macro_average_ROC_comparison.pdf'
    )

    print('All outputs generated successfully.')

if __name__ == '__main__':
    main()
