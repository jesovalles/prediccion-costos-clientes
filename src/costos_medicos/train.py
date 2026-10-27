"""Entrenamiento, comparación y selección reproducible de modelos."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .model import create_models

TARGET = "charges"


@dataclass
class TrainingResult:
    """Reúne el modelo final, su nombre, la tabla comparativa y métricas de prueba.

    Attributes:
        model: Pipeline seleccionado y reentrenado con todo el dataset.
        selected_model: Nombre del modelo que obtuvo menor MAE en validación.
        leaderboard: Métricas CV y de prueba de todos los modelos comparados.
        test_metrics: MAE, RMSE y R² en la partición de prueba del seleccionado.
    """

    model: Pipeline
    selected_model: str
    leaderboard: pd.DataFrame
    test_metrics: dict[str, float]


def _calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calcula MAE, RMSE y R² entre valores reales y estimados.

    MAE y RMSE quedan expresados en la unidad de ``charges`` (USD en el dataset).
    """
    residuals = actual.to_numpy() - predicted
    centered_actual = actual.to_numpy() - actual.mean()
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "r2": float(
            1 - np.sum(np.square(residuals)) / np.sum(np.square(centered_actual))
        ),
    }


def train_and_compare(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 5,
    search_iterations: int = 8,
) -> TrainingResult:
    """Evalúa modelos, selecciona por MAE CV y ajusta el seleccionado con todos los datos.

    Primero aparta una partición de prueba que no interviene en la selección.
    Compara seis regresores y una búsqueda aleatoria de Random Forest mediante
    K-fold sobre el conjunto de entrenamiento. Finalmente, reentrena el mejor
    pipeline con todos los registros disponibles para producir predicciones.

    Args:
        data: Dataset validado que incluye predictores y ``charges``.
        test_size: Proporción reservada para la evaluación final.
        random_state: Semilla que hace reproducible la división y los modelos.
        cv: Cantidad de pliegues utilizados en la validación cruzada.
        search_iterations: Combinaciones evaluadas en la búsqueda de Random Forest.

    Returns:
        El pipeline final, las métricas comparativas y los indicadores de prueba.

    Raises:
        ValueError: Si faltan columnas o hay filas/parámetros insuficientes.
    """
    if cv < 2:
        raise ValueError("La validación cruzada requiere al menos dos pliegues.")
    if search_iterations < 1:
        raise ValueError("La búsqueda de hiperparámetros requiere al menos una iteración.")
    if TARGET not in data.columns:
        raise ValueError(f"Falta la variable objetivo '{TARGET}'.")
    if len(data) < cv + 2:
        raise ValueError("No hay suficientes filas para la partición y validación cruzada.")

    # Se separa la variable objetivo de los predictores antes de cualquier ajuste.
    features = data.drop(columns=[TARGET])
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    # Los pliegues se mezclan y se fijan para repetir exactamente la evaluación.
    cross_validator = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    rows: list[dict[str, float | str]] = []
    fitted_candidates: dict[str, Pipeline] = {}

    # Todos los estimadores reciben exactamente los mismos pliegues y métricas.
    for name, pipeline in create_models(random_state).items():
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=1,
        )
        # El ajuste separado permite medir cada modelo en el mismo conjunto de prueba.
        pipeline.fit(x_train, y_train)
        test_scores = _calculate_metrics(y_test, pipeline.predict(x_test))
        rows.append(
            {
                "modelo": name,
                "mae_cv": float(-np.mean(scores["test_mae"])),
                "rmse_cv": float(-np.mean(scores["test_rmse"])),
                "r2_cv": float(np.mean(scores["test_r2"])),
                "mae_test": test_scores["mae"],
                "rmse_test": test_scores["rmse"],
                "r2_test": test_scores["r2"],
            }
        )
        fitted_candidates[name] = pipeline

    # Explora una muestra reproducible de combinaciones de hiperparámetros
    # para intentar mejorar el Random Forest de configuración fija.
    forest_pipeline = create_models(random_state)["Random Forest"]
    search = RandomizedSearchCV(
        estimator=forest_pipeline,
        param_distributions={
            "regresor__n_estimators": [100, 200, 300],
            "regresor__max_depth": [None, 8, 12, 16],
            "regresor__min_samples_leaf": [1, 2, 4],
            "regresor__max_features": [0.8, 1.0],
        },
        n_iter=search_iterations,
        scoring=scoring,
        refit="mae",
        cv=cross_validator,
        n_jobs=1,
        random_state=random_state,
    )
    search.fit(x_train, y_train)
    tuned_name = "Random Forest optimizado"
    tuned_scores = search.cv_results_
    best_index = search.best_index_
    tuned_test = _calculate_metrics(y_test, search.predict(x_test))
    rows.append(
        {
            "modelo": tuned_name,
            "mae_cv": float(-tuned_scores["mean_test_mae"][best_index]),
            "rmse_cv": float(-tuned_scores["mean_test_rmse"][best_index]),
            "r2_cv": float(tuned_scores["mean_test_r2"][best_index]),
            "mae_test": tuned_test["mae"],
            "rmse_test": tuned_test["rmse"],
            "r2_test": tuned_test["r2"],
        }
    )
    fitted_candidates[tuned_name] = search.best_estimator_

    # El MAE CV es el criterio de selección porque expresa el error absoluto
    # típico en dólares y es menos sensible a valores extremos que el RMSE.
    leaderboard = pd.DataFrame(rows).sort_values("mae_cv").reset_index(drop=True)
    selected_model = str(leaderboard.iloc[0]["modelo"])
    # El candidato se vuelve a ajustar con todos los datos para aprovecharlos
    # al máximo al generar predicciones futuras.
    final_model = clone(fitted_candidates[selected_model])
    final_model.fit(features, target)
    selected_test_metrics = leaderboard.iloc[0]

    return TrainingResult(
        model=final_model,
        selected_model=selected_model,
        leaderboard=leaderboard,
        test_metrics={
            "mae": float(selected_test_metrics["mae_test"]),
            "rmse": float(selected_test_metrics["rmse_test"]),
            "r2": float(selected_test_metrics["r2_test"]),
        },
    )
