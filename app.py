"""Interfaz Streamlit para explorar datos y estimar costos médicos."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.costos_medicos.data import load_data, validate_prediction_data
from src.costos_medicos.train import TrainingResult, train_and_compare

DATA_PATH = Path(__file__).resolve().parent / "data" / "insurance.csv"
# Estas son las únicas columnas que recibe el estimador; las columnas adicionales
# del CSV se conservan para que el usuario pueda identificar cada predicción.
VARIABLES_PREDICTORAS = ["age", "sex", "bmi", "children", "smoker", "region"]


@st.cache_resource(show_spinner="Comparando modelos y entrenando el seleccionado...")
def load_training_result() -> tuple[pd.DataFrame, TrainingResult]:
    """Carga los datos de referencia y entrena los modelos una sola vez."""
    # La caché evita repetir una comparación que incluye validación cruzada
    # cada vez que Streamlit vuelve a ejecutar el archivo de la aplicación.
    data = load_data(DATA_PATH)
    return data, train_and_compare(data)


def create_histogram(data: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    """Prepara los conteos de una variable numérica para graficar su distribución."""
    # Se limita el histograma a diez intervalos para que sea fácil de leer
    # incluso cuando el CSV tiene muchos valores distintos.
    bin_count = min(10, max(1, data[column].nunique()))
    groups = pd.cut(data[column], bins=bin_count, duplicates="drop")
    counts = groups.value_counts().sort_index()
    # Streamlit necesita categorías de texto para mostrar los intervalos en el eje.
    counts.index = counts.index.astype(str)
    return counts.rename_axis(label).rename("Registros").to_frame()


def show_model_metrics(result: TrainingResult) -> None:
    """Muestra indicadores de prueba, gráficas comparativas y tabla de modelos."""
    # MAE y RMSE se expresan en USD; R² es una medida relativa del ajuste.
    metric_columns = st.columns(3)
    metric_columns[0].metric("MAE en prueba", f"${result.test_metrics['mae']:,.0f} USD")
    metric_columns[1].metric(
        "RMSE en prueba",
        f"${result.test_metrics['rmse']:,.0f} USD",
    )
    metric_columns[2].metric("R² en prueba", f"{result.test_metrics['r2']:.3f}")

    st.caption(
        f"Modelo seleccionado por menor MAE promedio de validación cruzada: "
        f"**{result.selected_model}**."
    )
    # Se comparan tanto los resultados de validación cruzada como los de prueba.
    metric_left, metric_right = st.columns(2)
    with metric_left:
        st.markdown("#### Error absoluto medio (menor es mejor)")
        st.bar_chart(
            result.leaderboard.set_index("modelo")[["mae_cv", "mae_test"]],
            y_label="Error en USD",
        )
    with metric_right:
        st.markdown("#### Coeficiente R² (mayor es mejor)")
        st.bar_chart(
            result.leaderboard.set_index("modelo")[["r2_cv", "r2_test"]],
            y_label="R²",
        )

    tabla = result.leaderboard.rename(
        columns={
            "modelo": "Modelo",
            "mae_cv": "MAE CV (USD)",
            "rmse_cv": "RMSE CV (USD)",
            "r2_cv": "R² CV",
            "mae_test": "MAE prueba (USD)",
            "rmse_test": "RMSE prueba (USD)",
            "r2_test": "R² prueba",
        }
    )
    st.dataframe(
        tabla.style.format(
            {
                "MAE CV (USD)": "${:,.0f}",
                "RMSE CV (USD)": "${:,.0f}",
                "R² CV": "{:.3f}",
                "MAE prueba (USD)": "${:,.0f}",
                "RMSE prueba (USD)": "${:,.0f}",
                "R² prueba": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "CV corresponde a validación cruzada en los datos de entrenamiento. "
        "La prueba es una partición separada y no se utiliza para seleccionar modelos."
    )


st.set_page_config(
    page_title="Predicción de costos médicos",
    page_icon="💊",
    layout="wide",
)
st.title("Predicción de costos médicos")
st.write(
    "Explora los datos, compara modelos y estima costos para una persona "
    "o para múltiples registros desde un archivo CSV."
)

# El modelo se ajusta con el CSV del proyecto, nunca con el archivo de predicción
# cargado por el usuario. Si el conjunto de referencia no se puede leer, no se
# continúa para evitar mostrar estimaciones engañosas o incompletas.
try:
    reference_data, training = load_training_result()
except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
    st.error(f"No se pudo cargar o validar el conjunto de referencia: {error}")
    st.stop()

uploaded_file = st.sidebar.file_uploader(
    "Cargar CSV para el resumen y las predicciones",
    type=["csv"],
    help=(
        "Debe incluir age, sex, bmi, children, smoker y region. "
        "La columna charges es opcional."
    ),
)

# Las cargas de archivos se validan en cada ejecución de Streamlit. Se incluyen
# las categorías y rangos permitidos antes de enviar filas al estimador.
uploaded_data: pd.DataFrame | None = None
if uploaded_file is not None:
    try:
        uploaded_data = validate_prediction_data(pd.read_csv(uploaded_file))
        if uploaded_data.empty:
            st.sidebar.error("El CSV no contiene registros para analizar.")
            uploaded_data = None
        else:
            st.sidebar.success(
                f"CSV listo: {len(uploaded_data):,} registros validados."
            )
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.sidebar.error(f"No se pudo usar el CSV: {error}")

summary_data = uploaded_data if uploaded_data is not None else reference_data
# Si aún no hay archivo del usuario, el resumen muestra el dataset de entrenamiento.
summary_source = (
    "archivo CSV cargado"
    if uploaded_data is not None
    else "conjunto de referencia del proyecto"
)

summary_tab, individual_tab, batch_tab, models_tab = st.tabs(
    ["Resumen del dataset", "Predicción individual", "Predicción por CSV", "Modelos"]
)

# RESUMEN: los gráficos y promedios se calculan sobre el conjunto seleccionado,
# que puede ser el CSV del usuario o el conjunto de referencia.
with summary_tab:
    st.subheader("Resumen del dataset")
    st.caption(
        f"Fuente: {summary_source}. "
        f"Registros: {len(summary_data):,}. "
        f"Modelo predictivo entrenado con el conjunto de referencia."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Registros", f"{len(summary_data):,}")
    metric_columns[1].metric("Edad promedio", f"{summary_data['age'].mean():.1f} años")
    metric_columns[2].metric("IMC promedio", f"{summary_data['bmi'].mean():.1f}")
    if "charges" in summary_data:
        metric_columns[3].metric(
            "Costo promedio",
            f"${summary_data['charges'].mean():,.0f} USD",
        )
    else:
        metric_columns[3].metric(
            "Fumadores",
            f"{summary_data['smoker'].eq('yes').mean():.1%}",
        )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Distribución de edades")
        st.bar_chart(create_histogram(summary_data, "age", "Rango de edad"))
    with chart_right:
        st.markdown("#### Registros por región")
        st.bar_chart(
            summary_data["region"].value_counts().sort_index().rename("Registros"),
            y_label="Registros",
        )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Registros por hábito de fumar")
        st.bar_chart(
            summary_data["smoker"].value_counts().sort_index().rename("Registros"),
            y_label="Registros",
        )
    with chart_right:
        if "charges" in summary_data:
            st.markdown("#### Costo promedio por categoría")
            costo_por_categoria = summary_data.groupby(
                "smoker", observed=True
            )["charges"].mean()
            st.bar_chart(costo_por_categoria, y_label="Costo promedio (USD)")
        else:
            st.markdown("#### Distribución del IMC")
            st.bar_chart(create_histogram(summary_data, "bmi", "Rango de IMC"))

    st.markdown("#### Vista previa")
    st.dataframe(summary_data.head(20), use_container_width=True, hide_index=True)
    if uploaded_data is None:
        st.info(
            "Carga un CSV desde el panel lateral para ver su resumen y utilizarlo "
            "en la pestaña de predicción por archivo."
        )

# PREDICCIÓN INDIVIDUAL: el formulario crea un registro con las mismas variables
# que espera el pipeline de scikit-learn.
with individual_tab:
    st.subheader("Predicción individual")
    st.caption("La estimación es orientativa y no sustituye una cotización real.")
    with st.form("formulario_prediccion"):
        columna_izquierda, columna_derecha = st.columns(2)
        with columna_izquierda:
            age = st.number_input(
                "Edad",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
            )
            bmi = st.number_input(
                "Índice de masa corporal (IMC)",
                min_value=10.0,
                max_value=70.0,
                value=27.5,
                step=0.1,
            )
            children = st.number_input(
                "Número de hijos",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )
        with columna_derecha:
            sex = st.selectbox("Sexo registrado", ["female", "male"])
            smoker = st.selectbox("¿Fuma?", ["no", "yes"])
            region = st.selectbox(
                "Región",
                ["northeast", "northwest", "southeast", "southwest"],
            )
        submitted = st.form_submit_button("Calcular estimación", type="primary")

    if submitted:
        persona = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "children": children,
                    "smoker": smoker,
                    "region": region,
                }
            ]
        )
        # Se evita mostrar un costo negativo por posibles extrapolaciones del modelo.
        costo_estimado = max(float(training.model.predict(persona)[0]), 0)
        st.success(f"Costo médico anual estimado: **${costo_estimado:,.2f} USD**")

# PREDICCIÓN POR LOTES: se seleccionan solo las variables del modelo, pero se
# mantienen las demás columnas del archivo en la tabla y en la descarga.
with batch_tab:
    st.subheader("Predicción por archivo CSV")
    st.write(
        "Sube el archivo desde el panel lateral. El modelo requiere las columnas "
        "`age`, `sex`, `bmi`, `children`, `smoker` y `region`; `charges` es opcional."
    )
    if uploaded_data is None:
        st.info("Carga un CSV válido desde el panel lateral para habilitar la predicción.")
    else:
        prediction_rows = uploaded_data[VARIABLES_PREDICTORAS]
        predictions = training.model.predict(prediction_rows)
        results = uploaded_data.copy()
        # Agrega la estimación y limita valores por debajo de cero a cero.
        results["predicted_charges"] = predictions.clip(0)

        st.metric(
            "Costo promedio estimado",
            f"${results['predicted_charges'].mean():,.2f} USD",
        )
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar predicciones",
            data=results.to_csv(index=False).encode("utf-8-sig"),
            file_name="predicciones_costos_medicos.csv",
            mime="text/csv",
            type="primary",
        )
        # Si el archivo contiene la respuesta real, también se puede calcular el
        # error de estas predicciones; ese resultado no interviene en el entrenamiento.
        if "charges" in results:
            results["error_absoluto"] = (
                results["charges"] - results["predicted_charges"]
            ).abs()
            st.caption(
                "El CSV incluye cargos reales: el error absoluto medio de estos "
                f"registros es ${results['error_absoluto'].mean():,.2f} USD."
            )

# MODELOS: las métricas pertenecen a la evaluación hecha sobre los datos
# de referencia durante el entrenamiento, no al CSV cargado para predecir.
with models_tab:
    st.subheader("Métricas y comparación de modelos")
    show_model_metrics(training)
