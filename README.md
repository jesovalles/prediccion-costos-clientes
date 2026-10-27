# Predicción de costos médicos

Aplicación web en español para explorar datos de seguros médicos, comparar
modelos de regresión y estimar cargos anuales. Las estimaciones usan seis
características: edad (`age`), sexo (`sex`), índice de masa corporal (`bmi`),
número de hijos (`children`), hábito de fumar (`smoker`) y región (`region`).
El costo real a estimar se guarda en `charges`.

> **Importante:** es una estimación estadística basada en los datos disponibles;
> no es una cotización, asesoría médica ni garantía del costo real.

## Estructura del proyecto

```text
.
├── .streamlit/
│   └── config.toml                 # Configuración visual y de ejecución
├── data/
│   └── insurance.csv              # Datos de referencia para entrenar
├── eda/
│   └── Predicción de costos médicos.ipynb  # Análisis exploratorio original
├── src/
│   └── costos_medicos/
│       ├── __init__.py
│       ├── data.py                # Lectura, esquema y validación de datos
│       ├── model.py               # Transformaciones y algoritmos
│       └── train.py               # Comparación, evaluación y selección
├── tests/
│   └── test_costos_medicos.py     # Pruebas automatizadas
├── app.py                         # Interfaz Streamlit
├── requirements.txt               # Dependencias de Python
└── README.md
```

### Responsabilidad de cada módulo

- **`app.py`** construye la interfaz, permite cargar un CSV, presenta el resumen
  y las métricas, y solicita predicciones individuales o por lotes.
- **`data.py`** comprueba que los archivos contengan las columnas requeridas,
  convierte y valida los números, normaliza categorías y retiene identificadores
  u otras columnas adicionales de los archivos de predicción.
- **`model.py`** define el preprocesamiento y los modelos. Las transformaciones
  se ajustan dentro de cada pipeline para evitar filtraciones entre entrenamiento
  y evaluación.
- **`train.py`** reserva datos de prueba, compara algoritmos mediante validación
  cruzada, busca parámetros para Random Forest y selecciona el modelo por su
  error absoluto medio de validación.
- **`tests/`** comprueba la carga, validaciones y capacidad de generar
  predicciones.
- **`eda/`** contiene el notebook con análisis exploratorio, visualizaciones,
  correlaciones, agrupamiento y un análisis predictivo independiente.

## Requisitos

- Python 3.10 o posterior.
- `pip` para instalar los paquetes de `requirements.txt`.

## Instalar y ejecutar en Windows

Abre PowerShell o una terminal en la carpeta del proyecto y ejecuta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Si PowerShell impide activar el entorno virtual, puedes usar el ejecutable
directamente sin activar el entorno:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit muestra en la terminal la dirección local de la aplicación, por
ejemplo `http://localhost:8501`. Abre esa dirección en el navegador. Para detener
la aplicación, vuelve a la terminal y pulsa `Ctrl+C`.

### Ejecutar en macOS o Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## ¿Cómo funciona la aplicación?

### 1. Carga y entrenamiento

Al iniciar, la aplicación lee `data/insurance.csv`. Este es el **conjunto de
referencia** con el que se entrenan y comparan los modelos. El archivo debe
contener las seis variables predictoras y la columna objetivo `charges`. Los
registros duplicados exactos se eliminan al preparar el conjunto.

El entrenamiento se conserva en caché mientras el proceso de Streamlit siga
activo; navegar por las pestañas no vuelve a entrenar los modelos. Si cambias el
CSV de referencia, reinicia la aplicación o borra la caché desde el menú de
Streamlit para que se vuelva a cargar y entrenar.

### 2. Pestaña «Resumen del dataset»

Sin un archivo cargado, muestra estadísticas y gráficos del conjunto de
referencia. Al cargar un CSV válido, el resumen pasa a utilizar los datos de ese
archivo: cantidad de registros, edad e IMC promedio, regiones, hábito de fumar,
distribuciones y una vista previa.

Si el CSV contiene `charges`, también se muestra el costo promedio y el costo
promedio por hábito de fumar. Si no contiene esa columna, se presenta la
distribución del IMC en su lugar.

### 3. Pestaña «Predicción individual»

Completa los seis campos del formulario y pulsa **Calcular estimación**. La
aplicación prepara un único registro y lo envía al pipeline entrenado con el
conjunto de referencia.

### 4. Pestaña «Predicción por CSV»

Usa **Cargar CSV para el resumen y las predicciones**, en el panel lateral, y
después abre esta pestaña. Debe haber una fila por persona y estas columnas:

| Columna | Tipo esperado | Valores o ejemplo |
|---|---|---|
| `age` | Número | `35` |
| `sex` | Categoría | `female` o `male` |
| `bmi` | Número | `27.5` |
| `children` | Número entero | `1` |
| `smoker` | Categoría | `yes` o `no` |
| `region` | Categoría | `northeast`, `northwest`, `southeast` o `southwest` |
| `charges` | Número, opcional | Costo real conocido, por ejemplo `12000.50` |

Las categorías pueden escribirse con mayúsculas o espacios alrededor; se
normalizan antes de usarlas. Se validan los valores numéricos, los campos vacíos
y las categorías desconocidas. Si un dato requerido no es válido, la aplicación
muestra un mensaje en el panel lateral.

Por ejemplo, un archivo de entrada mínimo puede ser:

```csv
customer_id,age,sex,bmi,children,smoker,region
cliente-001,35,female,27.5,1,no,southeast
cliente-002,48,male,31.2,2,yes,northwest
```

La tabla de resultados y el botón de descarga agregan `predicted_charges` y
conservan `customer_id` y cualquier otra columna adicional. `charges` puede
incluirse cuando se conoce el valor real; en ese caso se calcula también el error
absoluto medio de esas filas. Los registros subidos **se usan para resumir y
predecir, no para entrenar ni seleccionar el modelo**.

### 5. Pestaña «Modelos»

Compara un predictor de referencia y seis algoritmos:

- Predicción constante del promedio.
- Regresión lineal.
- Random Forest.
- Extra Trees.
- HistGradientBoosting.
- Gradient Boosting.
- Random Forest con búsqueda aleatoria de hiperparámetros.

La pestaña muestra MAE, RMSE y R² en validación cruzada y en una partición de
prueba separada, además de gráficos comparativos. Se selecciona el menor MAE
promedio de validación cruzada; **las métricas de prueba no se usan para elegir
el algoritmo**. Tras evaluarlo, el pipeline seleccionado se vuelve a entrenar
con todos los datos de referencia para las predicciones de la aplicación.

## Conceptos de evaluación

- **MAE (error absoluto medio):** diferencia absoluta promedio entre el costo
  real y el estimado. Se expresa en dólares y es más fácil de interpretar cuando
  se quiere conocer el error típico.
- **RMSE (raíz del error cuadrático medio):** también está expresado en dólares,
  pero penaliza más los errores grandes.
- **R²:** mide cuánto varían los costos que explica el modelo; puede ser menor
  que cero si el modelo resulta peor que una predicción constante en esa prueba.
- **Validación cruzada (CV):** divide repetidamente los datos de entrenamiento
  en pliegues para estimar la estabilidad del modelo.
- **Conjunto de prueba:** partición separada que sirve como comprobación final
  de los candidatos y no interviene en la selección.

Los resultados dependen del dataset, la partición aleatoria y los parámetros.
Un buen resultado sobre este conjunto pequeño no garantiza el mismo desempeño
en otras poblaciones o periodos; conviene validar con datos externos antes de
usar las estimaciones en decisiones reales.

## Ejecutar las pruebas

Con el entorno virtual activo, ejecuta desde la raíz del proyecto:

```powershell
python -m pytest -q
```

Las pruebas comprueban el esquema y la eliminación de duplicados, aceptan un CSV
de predicción sin `charges`, rechazan categorías desconocidas y verifican que
los modelos puedan entrenarse y devolver una predicción.

## Preparar la publicación en GitHub

Revisa los archivos que vas a agregar y confirma que no incluyan credenciales,
archivos privados, entornos virtuales ni información que no deba publicarse.

El workflow de GitHub Actions en `.github/workflows/tests.yml` ejecuta las
pruebas en Python 3.10 y 3.12 para cada push a `main` y cada pull request hacia
`main`.

Desde PowerShell, en la raíz del proyecto, puedes preparar y revisar los cambios
antes de confirmar el commit:

```powershell
git status --short
git add .gitignore .github README.md app.py data eda requirements.txt src tests .streamlit
git diff --cached --check
git diff --cached --stat
git status
```

Revisa el resumen y el estado. Si todo es correcto, crea el commit y publícalo:

```powershell
git commit -m "Prepara aplicación de predicción de costos médicos"
git push origin main
```

No se han creado commits ni se ha enviado ningún cambio a GitHub.

## Abrir el análisis exploratorio

Abre `eda/Predicción de costos médicos.ipynb` en Visual Studio Code con las
extensiones de Python y Jupyter instaladas. Selecciona el entorno virtual del
proyecto como kernel e instala las dependencias de `requirements.txt`. Ejecuta
las celdas en orden desde el inicio: el notebook prepara la ruta a
`data/insurance.csv` y algunas celdas dependen de variables creadas previamente.

El notebook documenta el análisis exploratorio original; la aplicación Streamlit
implementa por separado la validación, comparación y selección reproducible de
modelos.
