"""
predict.py — Predicción de riesgo crediticio para nuevos clientes.

Requiere que train.py haya sido ejecutado previamente para generar:
  - ../models/decision_tree.pkl
  - ../models/label_encoder.pkl
  - ../models/feature_columns.pkl
"""

import sys
import os
import warnings

import pandas as pd
import joblib

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "decision_tree.pkl")
ENCODER_PATH  = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

# ── Carga de artefactos ───────────────────────────────────────────────────────
for path in [MODEL_PATH, ENCODER_PATH, FEATURES_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Archivo no encontrado: {path}\n"
            "Ejecuta primero 'python src/train.py' para generar los modelos."
        )

model           = joblib.load(MODEL_PATH)
le              = joblib.load(ENCODER_PATH)
feature_columns = joblib.load(FEATURES_PATH)


def predict(data: dict) -> str:
    """
    Predice el nivel de riesgo crediticio para un nuevo cliente.

    Parámetros
    ----------
    data : dict
        Diccionario con los campos del cliente. Los campos que coincidan
        con los usados en entrenamiento serán utilizados; el resto se
        completará con 0 automáticamente.

    Retorna
    -------
    str
        Nivel de riesgo: 'Alto', 'Medio' o 'Bajo'.
    """
    df = pd.DataFrame([data])

    # Aplicar get_dummies igual que en el preprocesamiento
    df = pd.get_dummies(df)

    # Alinear columnas con las del entrenamiento:
    #   - Columnas faltantes se agregan con valor 0
    #   - Columnas extra (no vistas en entrenamiento) se eliminan
    df = df.reindex(columns=feature_columns, fill_value=0)

    prediction_encoded = model.predict(df)
    prediction_label   = le.inverse_transform(prediction_encoded)
    return prediction_label[0]


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Cliente con perfil de bajo riesgo (historial limpio, ingresos altos)
    cliente_bajo_riesgo = {
        "loan_amnt":           5000,
        "funded_amnt":         5000,
        "funded_amnt_inv":     4975,
        "term":                36,
        "int_rate":            10.65,
        "installment":         162.87,
        "annual_inc":          80000,
        "dti":                 8.5,
        "delinq_2yrs":         0,
        "inq_last_6mths":      0,
        "open_acc":            8,
        "revol_bal":           5000,
        "revol_util":          20.0,
        "total_acc":           20,
        "emp_length":          10,
        "grade":               "A",
        "sub_grade":           "A1",
        "home_ownership":      "OWN",
        "verification_status": "Verified",
        "purpose":             "debt_consolidation",
        "mths_since_last_delinq": 60,
    }

    # Cliente con perfil de alto riesgo (deuda alta, incumplimientos recientes)
    cliente_alto_riesgo = {
        "loan_amnt":           1500000,
        "funded_amnt":         15000,
        "funded_amnt_inv":     15000,
        "term":                6,
        "int_rate":            22.0,
        "installment":         420.0,
        "annual_inc":          20000,
        "dti":                 35.0,
        "delinq_2yrs":         3,
        "inq_last_6mths":      6,
        "open_acc":            2,
        "revol_bal":           18000,
        "revol_util":          95.0,
        "total_acc":           5,
        "emp_length":          0,
        "grade":               "F",
        "sub_grade":           "F5",
        "home_ownership":      "RENT",
        "verification_status": "Not Verified",
        "purpose":             "small_business",
        "mths_since_last_delinq": 5,
    }

    print("=" * 45)
    print("   Predicción de Riesgo Crediticio")
    print("=" * 45)

    for nombre, cliente in [
        ("Cliente 1 (perfil favorable)",    cliente_bajo_riesgo),
        ("Cliente 2 (perfil desfavorable)", cliente_alto_riesgo),
    ]:
        resultado = predict(cliente)
        emoji = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}.get(resultado, "⚪")
        print(f"\n  {nombre}")
        print(f"  Riesgo predicho: {emoji} {resultado}")

    print("\n" + "=" * 45)