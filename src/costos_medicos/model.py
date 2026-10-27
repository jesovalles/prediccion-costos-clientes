"""Definición de preprocesamiento y modelos de regresión."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor

VARIABLES_NUMERICAS = ("age", "bmi", "children")
VARIABLES_CATEGORICAS = ("sex", "smoker", "region")


def create_preprocessor() -> ColumnTransformer:
    """Escala números y codifica categorías sin aprender de datos de prueba."""
    return ColumnTransformer(
        transformers=[
            # La escala es importante para modelos sensibles a la magnitud,
            # como la regresión lineal.
            ("numericas", StandardScaler(), list(VARIABLES_NUMERICAS)),
            (
                "categoricas",
                # One-hot evita imponer un orden numérico entre categorías.
                # handle_unknown permite inferencia aunque aparezca una categoría nueva.
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(VARIABLES_CATEGORICAS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def create_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Crea modelos reproducibles con el mismo preprocesamiento de entrada."""
    # La referencia predice el promedio; sirve para comprobar que los otros
    # modelos aportan valor por encima de una estimación trivial.
    estimadores = {
        "Referencia (promedio)": DummyRegressor(strategy="mean"),
        "Regresión lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random_state,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            loss="huber",
            random_state=random_state,
        ),
    }
    # El preprocesador forma parte de cada pipeline: en validación cruzada se
    # ajusta exclusivamente con el pliegue de entrenamiento, evitando filtraciones.
    return {
        nombre: Pipeline(
            steps=[
                ("preprocesamiento", create_preprocessor()),
                ("regresor", estimador),
            ]
        )
        for nombre, estimador in estimadores.items()
    }
