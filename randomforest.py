import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

# =====================
# CARGA Y PREPARACIÓN
# =====================
df = pd.read_csv("datos_acelerometro.csv")

le = LabelEncoder()
df['status'] = le.fit_transform(df['status'])  # 'Ansiedad' = 0, 'No ansiedad' = 1

X = df[['x', 'y', 'z']]
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================
# ENTRENAMIENTO
# =====================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)

# =====================
# CURVA DE APRENDIZAJE
# =====================
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    shuffle=True, random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

print("Tamaño del conjunto de entrenamiento\tAccuracy Entrenamiento\tAccuracy Validación")
for size, train_acc, test_acc in zip(train_sizes, train_scores_mean, test_scores_mean):
    print(f"{size:<35}{train_acc:.4f}\t\t\t{test_acc:.4f}")

# Gráfica curva de aprendizaje
plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Entrenamiento')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Validación')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='green')
plt.title('Curva de Aprendizaje')
plt.xlabel('Cantidad de datos de entrenamiento')
plt.ylabel('Precisión')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================
# PREDICCIONES Y MÉTRICAS
# =====================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n=== Métricas en Entrenamiento ===")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("F1-Score:", f1_score(y_train, y_train_pred))
print("ROC AUC:", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))

print("\n=== Métricas en Prueba ===")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1-Score:", f1_score(y_test, y_test_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# =====================
# MATRICES DE CONFUSIÓN
# =====================
cm_train = confusion_matrix(y_train, y_train_pred)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=le.classes_)
disp_train.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - Entrenamiento")
plt.show()

cm_test = confusion_matrix(y_test, y_test_pred)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=le.classes_)
disp_test.plot(cmap=plt.cm.Oranges)
plt.title("Matriz de Confusión - Prueba")
plt.show()

print("Matriz de Confusión - Entrenamiento:")
print(cm_train)

print("\nMatriz de Confusión - Prueba:")
print(cm_test)

# =====================
# CURVA ROC
# =====================
fpr_test, tpr_test, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label='Entrenamiento (AUC = {:.4f})'.format(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])), linestyle='--')
plt.plot(fpr_test, tpr_test, label='Prueba (AUC = {:.4f})'.format(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])), color='orange')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Curva ROC')
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
