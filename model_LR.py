import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# ===================== 1. 参数配置 ===================== #
feature_file = r"\data\merged_dbcan_matrix_2_re_lable_geneID.csv"
label_file = r"\data\group.txt"
output_dir = "lr_results"
os.makedirs(output_dir, exist_ok=True)

# ===================== 2. 加载数据 ===================== #
# 特征矩阵：行为特征，列为样本名 -> 需要转置
X = pd.read_csv(feature_file, index_col=0)
X = X.T  # 转置为 shape = [样本数, 特征数]

# 标签文件
group_df = pd.read_csv(label_file, sep="\t", header=None, index_col=0)
y = group_df[1]

# 对齐样本
X = X.loc[y.index]

# ===================== 3. 划分训练测试集 ===================== #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===================== 4. 多类逻辑回归训练 ===================== #
clf = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", max_iter=1000, class_weight="balanced"
)
clf.fit(X_train, y_train)

# ===================== 5. 模型预测与评估 ===================== #
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).T
report_df.to_csv(f"{output_dir}/classification_report.csv")
print(report_df)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
plt.close()

# ===================== 6. ROC 曲线绘制 ===================== #
# 标签二值化
classes = clf.classes_
y_test_bin = label_binarize(y_test, classes=classes)
y_score = clf.predict_proba(X_test)

# ROC 多分类
plt.figure(figsize=(6, 5))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/roc_curve.png", dpi=300)
plt.close()

# ===================== 7. 特征重要性分析 ===================== #
coef_df = pd.DataFrame(clf.coef_.T, index=X.columns, columns=classes)

# 每类 top20 特征
for cls in classes:
    top = coef_df[cls].abs().sort_values(ascending=False).head(20)
    top_features = coef_df.loc[top.index, cls]

    plt.figure(figsize=(6, 5))
    sns.barplot(x=top_features.values, y=top_features.index, palette="vlag")
    plt.title(f"Top 20 Features for {cls}")
    plt.xlabel("Logistic Regression Coefficient")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top20_features_{cls}.png", dpi=300)
    plt.close()

# 保存所有系数
coef_df.to_csv(f"{output_dir}/feature_coefficients.csv")
