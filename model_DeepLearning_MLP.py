import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             classification_report, roc_auc_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import accuracy_score

# 设置全局随机种子（确保实验可重复）
SEED = 42  # 可修改为其他固定值


# 初始化所有随机种子
def set_random_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)
    # 对于CUDA加速的GPU操作，需额外设置
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # 配置TensorFlow会话
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


set_random_seed(SEED)

# 定义统一的颜色映射
COLOR_MAPPING = {'NPA': '#BA5952', 'PAd': '#614390', 'PAc': '#008F93'}


# 1. 构建深度学习模型（已包含随机种子）
def build_deep_model(input_dim, num_classes):
    """
    构建深度神经网络模型（多层感知器MLP）。
    参数:
        input_dim (int): 输入特征维度（基因数）。
        num_classes (int): 分类数（组别数量）。
    返回:
        model (tf.keras.Model): 编译后的Keras模型。
    """
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.5, seed=SEED),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5, seed=SEED),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3, seed=SEED),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# 2. 绘制ROC曲线
def plot_auc(y_test, y_pred_prob, labels, color_mapping, output_file="roc_curve.pdf"):
    """
    计算并绘制每个类别的ROC曲线和AUC。
    """
    fpr = {}
    tpr = {}
    roc_auc = {}

    # 逐类计算ROC和AUC
    for i, label in enumerate(labels):
        y_true = (y_test == i).astype(int)
        y_scores = y_pred_prob[:, i]
        fpr[label], tpr[label], _ = roc_curve(y_true, y_scores)
        roc_auc[label] = auc(fpr[label], tpr[label])

    # 绘制ROC曲线
    plt.figure()
    for label in labels:
        plt.plot(fpr[label], tpr[label],
                 label=f'{label} (AUC = {roc_auc[label]:.2f})',
                 color=color_mapping.get(label, '#000000'))

    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(output_file, format='pdf')
    plt.close()
    print(f"ROC curve saved to {output_file}")


# 3. 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, group_mapping, output_file="confusion_matrix.pdf"):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    class_names = [group_mapping[i] for i in range(len(group_mapping))]

    # 计算各类别准确率和误判率
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    misclassification_per_class = 1 - accuracy_per_class

    # 创建带注释的矩阵
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm).astype(str)
    n_classes = cm.shape[0]

    for i in range(n_classes):
        for j in range(n_classes):
            # 显示数量和百分比
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    # 创建DataFrame
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(cm_df, annot=annot, fmt='',
                     cmap='Blues', cbar=False,
                     linewidths=0.5, linecolor='gray',
                     annot_kws={"size": 12, "va": 'center'})

    # 添加准确率/误判率标注
    for i, (acc, mis) in enumerate(zip(accuracy_per_class, misclassification_per_class)):
        ax.text(n_classes + 0.5, i + 0.5, f"Acc: {acc:.1%}\nMis: {mis:.1%}",
                ha='left', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 美化图表
    plt.title("Confusion Matrix with Accuracy/Misclassification Rates", pad=20, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel(f'Predicted Label\n\nOverall Accuracy: {accuracy_score(y_true, y_pred):.2%}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 调整布局
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Enhanced confusion matrix saved to {output_file}")

def load_normalized_data(normalized_gene_file, group_file):
    # 加载标准化数据
    normalized_df = pd.read_csv(normalized_gene_file, index_col=0)
    gene_names = normalized_df.index.tolist()
    sample_names = normalized_df.columns.tolist()
    X_scaled = normalized_df.values
    print(f"原始数据维度: {normalized_df.shape} (文件: {normalized_gene_file})")

    # 加载分组数据
    group_data = pd.read_csv(group_file, sep='\t', header=None, names=['sample', 'group'])
    group_data = group_data.set_index('sample')
    group_data = group_data.loc[sample_names]  # 对齐样本顺序

    # 检查并报告NaN
    # nan_count = np.isnan(X_scaled).sum()
    # if nan_count > 0:
    #     print(f"Warning: Data contains {nan_count} NaN values. These will be imputed with column means.")

    # 标签编码（固定顺序）
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(group_data['group'].values)
    group_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print("Group Mapping:", group_mapping)

    return X_scaled.T, y, group_mapping, gene_names  # 转置为样品×基因


# 5. 提取特征基因（一对多随机森林 + 卡方检验）
def extract_feature_genes_one_vs_rest_rf(
        X_train, y_train,
        gene_names,
        group_mapping,
        top_n=15,
        output_file="gene_importance_one_vs_rest_rf.txt",
        threshold=0.0
):
    """
    对每个组别执行一对多（One-vs-Rest）策略：
    1) 使用随机森林评估特征重要性
    2) 使用卡方检验（Chi-Squared Test）评估统计显著性
    3) 以随机森林的重要性排序，并将前 top_n 个基因 (基因名, importance, p-value) 写入文件

    参数:
        X_train (numpy.ndarray): 训练数据（样品数 × 基因数），此时已是标准化后的数值.
        y_train (numpy.ndarray): 训练集标签.
        gene_names (list): 基因名称列表.
        group_mapping (dict): 标签索引到组别名称的映射.
        top_n (int): 每个组别提取的特征基因数量.
        output_file (str): 保存特征基因的文件路径.
        threshold (float): 当标准化后数值 > threshold 时视为presence(1)，否则absence(0).

    返回:
        gene_importance (dict):
          { group_name: [(gene_name, importance, p_value), ...], ... }
    """
    gene_importance = {}
    unique_groups = np.unique(y_train)

    # 定义插补器（处理可能的 NaN）
    imputer = SimpleImputer(strategy='mean')

    with open(output_file, 'w') as f:
        for group in unique_groups:
            group_name = group_mapping.get(group, 'Unknown')
            # 创建二分类标签: 当前组别 vs 其他组别
            binary_y = (y_train == group).astype(int)

            # ---- 1) 随机森林的重要性 ----
            pipeline = Pipeline([
                ('imputer', imputer),
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    random_state=SEED,  # 固定随机森林种子
                    class_weight='balanced'
                ))
            ])

            # 训练随机森林
            try:
                pipeline.fit(X_train, binary_y)
                importances = pipeline.named_steps['rf'].feature_importances_
            except ValueError as e:
                print(f"Error training RF for group {group_name}: {e}")
                importances = np.zeros(X_train.shape[1])

            # ---- 2) 卡方检验(p值) ----
            p_values = []
            for gene_idx in range(X_train.shape[1]):
                # 对标准化后的数值, 做 presence/absence
                gene_values = X_train[:, gene_idx]
                # 例如 threshold=0.0: 大于0视为presence(1)，否则absence(0)
                gene_presence = (gene_values > threshold).astype(int)

                # 构建 2×2 列联表
                contingency_table = pd.crosstab(gene_presence, binary_y)
                # 确保表格完整
                contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                try:
                    _, p, _, _ = chi2_contingency(contingency_table)
                except:
                    p = 1.0
                p_values.append(p)

            p_values = np.array(p_values)

            # ---- 3) 排序并选取 top_n ----
            # 以随机森林importance从高到低排序
            top_indices = np.argsort(importances)[::-1][:top_n]
            top_info = []
            for idx in top_indices:
                top_info.append(
                    (gene_names[idx], importances[idx], p_values[idx])
                )

            # 保存到字典
            gene_importance[group_name] = top_info

            # ---- 写入文件 ----
            f.write(f"Group {group} ({group_name}):\n")
            f.write(f"Top {top_n} Feature Genes (RF Importance + Chi2 p-value), threshold={threshold}:\n")
            for gene_name, imp, p_val in top_info:
                f.write(f"{gene_name}\tImportance={imp:.4f}\tp={p_val:.4e}\n")
            f.write("\n")

    print(f"Top {top_n} feature genes per group saved to {output_file}")
    return gene_importance


# 5. 绘制特征基因重要性排序图（一对多随机森林 + p值）
def plot_gene_importance_one_vs_rest_rf(
        gene_importance,
        output_file="gene_importance_one_vs_rest_rf.pdf"
):
    """
    绘制每个组别的前 top_n 特征基因条形图（X轴显示随机森林Importance）。
    p值可在图上加注释，但此处仅以文本形式保存到txt文件中。
    """
    num_groups = len(gene_importance)
    if num_groups == 0:
        print("No gene importance data to plot.")
        return

    fig, axs = plt.subplots(1, num_groups, figsize=(6 * num_groups, 6), constrained_layout=True)

    if num_groups == 1:
        axs = [axs]

    for ax, (group, info_list) in zip(axs, gene_importance.items()):
        gene_names = [item[0] for item in info_list]
        importances = [item[1] for item in info_list]

        ax.barh(gene_names[::-1], importances[::-1], color=COLOR_MAPPING.get(group, '#000000'))
        ax.set_title(f"{group} Top Genes")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Gene")

    plt.savefig(output_file, format='pdf')
    plt.close()
    print(f"Gene importance plot saved to {output_file}")


# 7. 主函数（所有随机操作均设置种子）
def deep_learning_classification():
    # 数据路径
    normalized_gene_file = r'\data\merged_dbcan_matrix_2_re_lable_geneID.csv'
    group_file = r'\data\group.txt'

    # 加载标准化数据、标签
    X_scaled_T, y, group_mapping, gene_names = load_normalized_data(normalized_gene_file, group_file)

    # 数据集划分（固定种子）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_T, y,
        test_size=0.3,
        random_state=SEED,  # 固定划分
        stratify=y
    )

    # 计算类权重
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    # 构建模型（已内置种子）
    model = build_deep_model(X_train.shape[1], len(group_mapping))

    # 训练（回调函数中也避免随机性）早停、lr衰减
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ],
        class_weight=class_weights_dict,
        shuffle=True,  # Keras默认会shuffle，但已通过SEED固定
        verbose=1
    )

    # 评估和预测
    # 评估
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 预测
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 绘制ROC曲线
    plot_auc(y_test, y_pred_prob, list(group_mapping.values()), COLOR_MAPPING, "roc_curve.pdf")

    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred, group_mapping, "confusion_matrix.pdf")

    # 输出分类报告（避免零除警告）
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(group_mapping.values()), zero_division=0))

    # 基于随机森林+卡方检验提取特征基因
    # 这里 threshold=0.0, 表示 >0 即视为presence.
    # 您可根据数据分布自定义阈值，如0.5或-0.2等。
    # 特征基因提取（随机森林设置种子）

    gene_importance_rf = extract_feature_genes_one_vs_rest_rf(
        X_train,
        y_train,
        gene_names=gene_names,
        group_mapping=group_mapping,
        top_n=15,  # 可自行调整
        output_file="gene_importance_one_vs_rest_rf.txt",
        threshold=0.0
    )

    # 绘制特征基因重要性排序图
    if not gene_importance_rf:
        print("Warning: gene_importance_rf is empty. Skip plotting.")
    else:
        plot_gene_importance_one_vs_rest_rf(gene_importance_rf, "gene_importance_one_vs_rest_rf.pdf")

    # 绘制训练过程
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.pdf", format='pdf')
    plt.close()
    print("Training history saved to training_history.pdf")


if __name__ == '__main__':
    deep_learning_classification()