"""Pruebas de carga, comparación de modelos y predicción."""

from pathlib import Path

import pandas as pd
import pytest

from src.costos_medicos.data import (
    COLUMNAS,
    load_data,
    validate_prediction_data,
)
from src.costos_medicos.train import train_and_compare

ROOT = Path(__file__).resolve().parents[1]


def test_load_data_validates_and_removes_duplicate_rows() -> None:
    """Comprueba que el CSV incluido se carga con el esquema y tamaño esperados."""
    data = load_data(ROOT / "data" / "insurance.csv")

    assert tuple(data.columns) == COLUMNAS
    assert len(data) == 1337
    assert not data.duplicated().any()


def test_load_data_rejects_missing_required_columns(tmp_path: Path) -> None:
    """Evita entrenar con un archivo que no contiene todas las variables."""
    csv_path = tmp_path / "incompleto.csv"
    pd.DataFrame({"age": [30], "charges": [1000]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        load_data(csv_path)


def test_validate_prediction_data_accepts_predictors_without_target() -> None:
    """Permite usar CSV nuevos que todavía no tienen cargos reales."""
    data = pd.DataFrame(
        [
            {
                "age": 35,
                "sex": "Female",
                "bmi": 27.5,
                "children": 1,
                "smoker": "NO",
                "region": "Southeast",
                "customer_id": "cliente-1",
            }
        ]
    )

    validated = validate_prediction_data(data)

    assert len(validated) == 1
    assert validated.loc[0, "sex"] == "female"
    assert validated.loc[0, "smoker"] == "no"
    assert validated.loc[0, "region"] == "southeast"
    assert validated.loc[0, "customer_id"] == "cliente-1"
    assert "charges" not in validated


def test_validate_prediction_data_rejects_unknown_categories() -> None:
    """Evita generar resultados si los valores categóricos no son reconocidos."""
    data = pd.DataFrame(
        [
            {
                "age": 35,
                "sex": "other",
                "bmi": 27.5,
                "children": 1,
                "smoker": "no",
                "region": "southeast",
            }
        ]
    )

    with pytest.raises(ValueError, match="Categorías no reconocidas"):
        validate_prediction_data(data)


def test_model_comparison_returns_a_fitted_model_and_metrics() -> None:
    """Verifica que se comparen alternativas y el resultado permita predecir."""
    data = load_data(ROOT / "data" / "insurance.csv").sample(
        n=100,
        random_state=7,
    )
    result = train_and_compare(data, cv=3, search_iterations=2)
    ejemplo = data.drop(columns=["charges"]).iloc[[0]]
    prediction = result.model.predict(ejemplo)

    assert len(result.leaderboard) == 7
    assert result.selected_model == result.leaderboard.iloc[0]["modelo"]
    assert set(result.test_metrics) == {"mae", "rmse", "r2"}
    assert all(value >= 0 for key, value in result.test_metrics.items() if key != "r2")
    assert len(prediction) == 1
