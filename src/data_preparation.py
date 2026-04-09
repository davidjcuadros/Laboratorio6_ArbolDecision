import pandas as pd
import re


# Columnas con data leakage (información posterior al otorgamiento)
LEAKAGE_COLS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "last_pymnt_d", "last_pymnt_amnt",
    "recoveries", "collection_recovery_fee", "last_credit_pull_d",
    "out_prncp", "out_prncp_inv",
]

# Columnas sin valor analítico o identificadores
IRRELEVANT_COLS = [
    "url", "member_id", "desc", "title", "zip_code",
]

# Columnas de texto libre / fechas no estructuradas que no se usarán
TEXT_DATE_COLS = [
    "issue_d", "earliest_cr_line", "emp_title", "addr_state",
]


def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV separado por ; con encoding estándar."""
    return pd.read_csv(path, sep=";", low_memory=False)


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia columnas que vienen como string con formato sucio:
    - int_rate / revol_util: '10.65%' -> 10.65
    - term: ' 36 months' -> 36
    - Números con puntos como separador de miles europeo (e.g. '5.863.155.187')
      que realmente son decimales con punto -> se toman solo si tienen un único punto.
    """
    # Porcentajes
    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Plazo del préstamo
    if "term" in df.columns:
        df["term"] = (
            df["term"].astype(str)
            .str.extract(r"(\d+)")[0]
        )
        df["term"] = pd.to_numeric(df["term"], errors="coerce")

    # emp_length: '10+ years' -> 10, '< 1 year' -> 0
    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"].astype(str)
            .str.replace("< 1 year", "0", regex=False)
            .str.replace("10+ years", "10", regex=False)
            .str.extract(r"(\d+)")[0]
        )
        df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

    return df


def validate_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    El CSV ya tiene loan_status con valores 'Alto', 'Medio', 'Bajo'.
    Solo renombramos la columna a 'riesgo' y validamos que no haya
    valores inesperados.
    """
    valid_values = {"Alto", "Medio", "Bajo"}
    if "loan_status" not in df.columns:
        raise ValueError("La columna 'loan_status' no existe en el dataset.")

    unexpected = set(df["loan_status"].dropna().unique()) - valid_values
    if unexpected:
        raise ValueError(
            f"Valores inesperados en loan_status: {unexpected}. "
            f"Se esperaban: {valid_values}. "
            "Verifica que el CSV ya haya sido pre-procesado."
        )

    df = df.rename(columns={"loan_status": "riesgo"})
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas de data leakage, irrelevantes y de texto libre."""
    all_to_drop = LEAKAGE_COLS + IRRELEVANT_COLS + TEXT_DATE_COLS
    df = df.drop(columns=[c for c in all_to_drop if c in df.columns])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica get_dummies a las columnas categóricas restantes,
    excluyendo la variable objetivo.
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "riesgo" in cat_cols:
        cat_cols.remove("riesgo")
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
    return df


def preprocess(path: str) -> pd.DataFrame:
    """Pipeline completo de preparación de datos."""
    df = load_data(path)
    df = validate_target(df)       # renombra loan_status -> riesgo y valida
    df = drop_columns(df)          # elimina leakage e irrelevantes
    df = clean_numeric_columns(df) # limpia porcentajes, plazos, etc.
    df = encode_categoricals(df)   # dummies para categóricas
    df = df.dropna(subset=["riesgo"])  # elimina filas sin etiqueta
    return df