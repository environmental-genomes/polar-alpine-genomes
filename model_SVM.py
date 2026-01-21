import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# ===================== 1. 参数配置 ===================== #
feature_file = r"merged_dbcan_matrix_2_re_lable_geneID.csv"
label_file = r"group.txt"
output_dir = r"svm_results"
os.makedirs(output_dir, exist_ok=True)

pdf_path = os.path.join(output_dir, "svm_results.pdf")

# ===================== 2. 加载数据 ===================== #
# 特征矩阵（行为特征，列为样本）-> 转置为 样本 × 特征
X = pd.read_csv(feature_file, index_col=0)
X = X.T

# 标签文件（index 为样本名）
group_df = pd.read_csv(label_file, sep="\t", header=None, index_col=0)
y = group_df[1]

# 按标签文件顺序对齐样本
X = X.loc[y.index]

# ===================== 3. 标签编码 & 数据标准化 ===================== #
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 转换为 0, 1, 2

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===================== 4. 划分训练/测试集 ===================== #
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ===================== 5. SVM + 网格搜索 ===================== #
param_grid = {
    "estimator__C": [0.1, 1, 10],
    "estimator__gamma": ["scale", 0.1, 0.01],
    "estimator__kernel": ["rbf", "linear"]
}

svc = OneVsRestClassifier(SVC(probability=True, random_state=42))

grid = GridSearchCV(
    svc, param_grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

best_params = grid.best_params_
print("Best parameters:", best_params)
model = grid.best_estimator_

# ===================== 6. 模型预测与评估 ===================== #
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
clf_report = classification_report(y_test, y_pred, target_names=le.classes_)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nClassification report:\n", clf_report)
print("\nConfusion matrix:\n", cm)

# ===================== 7. 保存到 PDF ===================== #
with PdfPages(pdf_path) as pdf:
    # ROC 曲线
    y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    y_score = model.decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(6, 6))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{le.classes_[i]} (AUC={roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SVM Multiclass ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    pdf.savefig()  # 保存当前图到 PDF
    plt.close()

    # 混淆矩阵热图
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # 文本结果页
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    text_str = f"Best Parameters:\n{best_params}\n\n" \
               f"Accuracy: {acc:.4f}\n\n" \
               f"Classification Report:\n{clf_report}\n" \
               f"Confusion Matrix:\n{cm}"
    ax.text(0, 1, text_str, fontsize=10, va="top", family="monospace")
    pdf.savefig()
    plt.close()

print(f"所有结果已保存到: {pdf_path}")
