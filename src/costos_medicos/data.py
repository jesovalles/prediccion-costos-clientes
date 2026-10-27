"""Carga y validación del conjunto de costos médicos."""

from pathlib import Path

import numpy as np
import pandas as pd

VARIABLES_PREDICTORAS = ("age", "sex", "bmi", "children", "smoker", "region")
# El dataset de entrenamiento debe incluir la variable objetivo charges.
COLUMNAS = (*VARIABLES_PREDICTORAS, "charges")
# Los valores categóricos válidos coinciden con el dataset de referencia del proyecto.
VARIABLES_NUMERICAS = ("age", "bmi", "children", "charges")
CATEGORIAS_VALIDAS = {
    "sex": {"female", "male"},
    "smoker": {"yes", "no"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
}


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Lee el CSV, comprueba su esquema y devuelve filas válidas sin duplicados."""
    data = pd.read_csv(csv_path)
    # Para entrenar siempre se necesita tanto el conjunto de predictores
    # como la columna de costos reales que el modelo debe aprender a estimar.
    columnas_faltantes = set(COLUMNAS).difference(data.columns)
    if columnas_faltantes:
        faltantes = ", ".join(sorted(columnas_faltantes))
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}.")

    # Se descartan columnas ajenas al entrenamiento y se comparte la validación
    # de formato usada también para los CSV nuevos de predicción.
    data = validate_prediction_data(data.loc[:, COLUMNAS])
    if "charges" not in data:
        raise ValueError("Falta la variable objetivo 'charges' en el conjunto.")
    return data.drop_duplicates().reset_index(drop=True)


def validate_prediction_data(data: pd.DataFrame) -> pd.DataFrame:
    """Valida variables predictoras y conserva cargos cuando el CSV los incluye."""
    # Para estimar no es necesario conocer charges; las seis características sí.
    columnas_faltantes = set(VARIABLES_PREDICTORAS).difference(data.columns)
    if columnas_faltantes:
        faltantes = ", ".join(sorted(columnas_faltantes))
        raise ValueError(f"Faltan columnas requeridas para predecir: {faltantes}.")

    # Se conserva una copia para no modificar el DataFrame que entrega el llamador.
    # Así también se retienen identificadores u otras columnas informativas.
    valid_data = data.copy()

    # Convierte texto numérico a números y comunica claramente los formatos inválidos.
    for columna in VARIABLES_NUMERICAS:
        if columna in valid_data:
            try:
                valid_data[columna] = pd.to_numeric(valid_data[columna], errors="raise")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"La columna '{columna}' debe contener solo números."
                ) from error

    # Solo se validan como obligatorias las variables predictoras y, si existe,
    # el costo real; una columna adicional puede quedar vacía sin impedir predecir.
    required_columns = list(VARIABLES_PREDICTORAS)
    if "charges" in valid_data:
        required_columns.append("charges")
    if valid_data[required_columns].isna().any().any():
        raise ValueError("El CSV contiene valores vacíos en columnas requeridas.")
    numeric_columns = [
        columna for columna in VARIABLES_NUMERICAS if columna in valid_data
    ]
    # NaN, infinito y menos infinito no son entradas utilizables para el modelo.
    if not np.isfinite(valid_data[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Las columnas numéricas deben contener valores finitos.")
    if (valid_data[["age", "children"]] < 0).any().any():
        raise ValueError("La edad y el número de hijos no pueden ser negativos.")
    if (valid_data["bmi"] <= 0).any():
        raise ValueError("El IMC debe ser mayor que cero.")
    if "charges" in valid_data and (valid_data["charges"] < 0).any():
        raise ValueError("Los cargos no pueden ser negativos.")

    # Elimina diferencias de mayúsculas y espacios antes de comparar categorías.
    for columna, categorias in CATEGORIAS_VALIDAS.items():
        observadas = set(valid_data[columna].astype(str).str.strip().str.lower())
        desconocidas = observadas.difference(categorias)
        if desconocidas:
            valores = ", ".join(sorted(desconocidas))
            raise ValueError(f"Categorías no reconocidas en '{columna}': {valores}.")
        valid_data[columna] = (
            valid_data[columna].astype(str).str.strip().str.lower()
        )

    return valid_data.reset_index(drop=True)
