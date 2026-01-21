import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score

# 固定随机种子确保复现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
import random
random.seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 颜色映射
COLOR_MAPPING = {'NPA': '#BA5952', 'PAd': '#614390', 'PAc': '#008F93'}

def load_normalized_data(normalized_gene_file, group_file):
    normalized_df = pd.read_csv(normalized_gene_file, index_col=0)
    gene_names = normalized_df.index.tolist()
    sample_names = normalized_df.columns.tolist()
    X_scaled = normalized_df.values
    print(f"Data shape: {normalized_df.shape}")

    group_data = pd.read_csv(group_file, sep='\t', header=None, names=['sample', 'group'])
    group_data = group_data.set_index('sample')
    group_data = group_data.loc[sample_names]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(group_data['group'].values)
    group_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print("Group mapping:", group_mapping)

    return X_scaled.T, y, group_mapping, gene_names  # 样本 x 特征

def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5, seed=SEED),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5, seed=SEED),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3, seed=SEED),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_roc(y_true, y_score, class_names, colors, output_path):
    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure()
    for i, label in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        fpr[label], tpr[label], _ = roc_curve(y_bin, y_score[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])
        plt.plot(fpr[label], tpr[label], color=colors.get(label, 'black'),
                 label=f"{label} (AUC={roc_auc[label]:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve saved: {output_path}")

def plot_confusion_matrix(y_true, y_pred, group_mapping, output_path):
    cm = confusion_matrix(y_true, y_pred)
    class_names = [group_mapping[i] for i in range(len(group_mapping))]

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved: {output_path}")

def integrated_gradients(model, inputs, baseline=None, target_class_idx=None, m_steps=50):
    if baseline is None:
        baseline = tf.zeros(shape=inputs.shape)

    # Scale inputs and compute gradients.
    scaled_inputs = [baseline + (float(i) / m_steps) * (inputs - baseline) for i in range(0, m_steps + 1)]
    grads = []

    for x in scaled_inputs:
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = model(x)
            if target_class_idx is None:
                output = tf.reduce_max(preds, axis=1)
            else:
                output = preds[:, target_class_idx]
        grad = tape.gradient(output, x)
        grads.append(grad)

    grads = tf.stack(grads)  # (m_steps+1, batch_size, features)
    avg_grads = tf.reduce_mean(grads[:-1] + grads[1:], axis=0) / 2.0

    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads.numpy()

def main():
    normalized_gene_file = r"\data\merged_dbcan_matrix_2_re_lable_geneID.csv"
    group_file = r"\data\group.txt"
    result_dir = "mlp_results1"
    os.makedirs(result_dir, exist_ok=True)

    X, y, group_mapping, gene_names = load_normalized_data(normalized_gene_file, group_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    model = build_model(X_train.shape[1], len(group_mapping))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        validation_data=(X_test, y_test),
                        class_weight=class_weights_dict,
                        shuffle=True,
                        callbacks=callbacks,
                        verbose=2)

    # 评估
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    # 预测概率和类别
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 保存训练历史图
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(result_dir, "training_history.pdf"))
    plt.close()

    # ROC 曲线和混淆矩阵
    plot_roc(y_test, y_pred_prob, list(group_mapping.values()), COLOR_MAPPING, os.path.join(result_dir, "roc_curve.pdf"))
    plot_confusion_matrix(y_test, y_pred, group_mapping, os.path.join(result_dir, "confusion_matrix.pdf"))

    # 分类报告
    print(classification_report(y_test, y_pred, target_names=list(group_mapping.values()), zero_division=0))

    # 计算 Integrated Gradients 特征重要性（用测试集前200个样本）
    X_test_sub = X_test[:200]
    ig_attributions = []
    for class_idx in range(len(group_mapping)):
        ig = integrated_gradients(model, tf.convert_to_tensor(X_test_sub, dtype=tf.float32),
                                  baseline=tf.zeros_like(X_test_sub, dtype=tf.float32),
                                  target_class_idx=class_idx)
        ig_attributions.append(np.mean(ig, axis=0))

    ig_attributions = np.array(ig_attributions)  # shape (num_classes, num_features)

    # 画每类特征重要性bar图
    for i, class_name in enumerate(group_mapping.values()):
        plt.figure(figsize=(10,6))
        indices = np.argsort(ig_attributions[i])[::-1][:30]
        plt.bar(range(30), ig_attributions[i][indices], tick_label=[gene_names[j] for j in indices])
        plt.xticks(rotation=90)
        plt.title(f"Integrated Gradients - Top 30 features for {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"IG_feature_importance_{class_name}.pdf"))
        plt.close()

    # 画热图显示所有类对所有特征的重要性（只显示top50特征）
    top50_indices = np.argsort(np.mean(ig_attributions, axis=0))[::-1][:50]
    plt.figure(figsize=(12,6))
    sns.heatmap(ig_attributions[:, top50_indices], xticklabels=[gene_names[i] for i in top50_indices],
                yticklabels=list(group_mapping.values()), cmap='RdBu_r', center=0)
    plt.title("Integrated Gradients Feature Importance Heatmap (Top 50 features)")
    plt.savefig(os.path.join(result_dir, "IG_feature_importance_heatmap.pdf"))
    plt.close()

    print(f"Results saved to directory: {result_dir}")

if __name__ == "__main__":
    main()
