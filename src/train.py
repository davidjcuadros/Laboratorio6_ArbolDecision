
import sys
import os

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Permite importar data_preparation desde el mismo directorio src/
sys.path.insert(0, os.path.dirname(__file__))
from data_preparation import preprocess

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataProccesingLoans.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Carga y preparación ───────────────────────────────────────────────────────
print(f"[1/5] Cargando datos desde: {DATA_PATH}")
df = preprocess(DATA_PATH)

print(f"      Registros: {len(df):,}  |  Columnas: {len(df.columns)}")
print(f"      Distribución de riesgo:\n{df['riesgo'].value_counts()}\n")

# ── Features / Target ─────────────────────────────────────────────────────────
X = df.drop("riesgo", axis=1)
y = df["riesgo"]

# Guardar columnas del entrenamiento para usarlas en predict.py
feature_columns = X.columns.tolist()

# Codificar target (Alto=0, Bajo=1, Medio=2 según orden alfabético)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"[2/5] Clases del LabelEncoder: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

# ── División train/test ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"[3/5] Train: {len(X_train):,} muestras  |  Test: {len(X_test):,} muestras\n")

# ── Modelo ────────────────────────────────────────────────────────────────────
print("[4/5] Entrenando DecisionTreeClassifier (max_depth=5)...")
model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# ── Evaluación ────────────────────────────────────────────────────────────────
print("\n[5/5] Evaluación en conjunto de prueba:")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {acc:.4f}")

print("\n  Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("  Matriz de confusión:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df)

# ── Guardar artefactos ────────────────────────────────────────────────────────
model_path    = os.path.join(MODEL_DIR, "decision_tree.pkl")
encoder_path  = os.path.join(MODEL_DIR, "label_encoder.pkl")
features_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

joblib.dump(model,           model_path)
joblib.dump(le,              encoder_path)
joblib.dump(feature_columns, features_path)

print(f"\n  Modelo guardado en   : {model_path}")
print(f"  Encoder guardado en  : {encoder_path}")
print(f"  Features guardadas en: {features_path}")