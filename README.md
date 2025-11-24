# Selección de variables con Algoritmo Genético + Regresión Logística (Deserción Temprana)

Este repositorio contiene un script en Python para **selección automática de variables** usando un **Algoritmo Genético (SGA)** y evaluación con **Regresión Logística** (Scikit‑Learn).  
Está adaptado a un proyecto de investigación sobre **Modelo predictivo para identificar estudiantes en riesgo de deserción temprana en educación superior**.

> Autor del repositorio y adaptación: *Favián Flores Lira*  
> Contexto: Doctorado en Tecnologías de la Información, UDG/CUCEA.

---

## Qué hace el script?

1. Carga un archivo CSV con datos reales de estudiantes.
2. Define un conjunto de variables candidatas.
3. Representa cada subconjunto de variables como un cromosoma binario.
4. Optimiza el subconjunto buscando maximizar el desempeño del modelo (AUC o Accuracy).
5. Genera dos gráficas:
   - Evolución del fitness por generación
   - Variables seleccionadas por el mejor individuo

---

---

## Nota sobre la implementación (autoría e inspiración)

Este repositorio implementa un **Algoritmo Genético estándar para selección de variables**, un enfoque ampliamente documentado y usado en Machine Learning.  
La lógica general sigue prácticas comunes en feature selection con GA: cromosomas binarios 0/1, selección por torneo, cruce de un punto, mutación bit a bit, elitismo y evaluación con validación cruzada.

La **adaptación al contexto de deserción temprana universitaria**, la selección de variables candidatas, el preprocesamiento, la configuración de métricas y la estructura final del código fueron realizados por **Favián Flores Lira** como parte de su proyecto doctoral.

Se utilizan librerías open‑source (NumPy, Pandas, Scikit‑Learn, Matplotlib) bajo licencias permisivas. Este repositorio no redistribuye dichas librerías; únicamente las emplea como dependencias.

---

## Estructura

```
.
├── ga_seleccion_variables_v2.py   # Script principal
├── datos_desercion.csv            # (NO incluido) tu dataset
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── .gitignore
```

---

## Requisitos

Python 3.9+ recomendado.

Instala dependencias:

```bash
pip install -r requirements.txt
```

---

## Uso rápido

1. Coloca tu dataset como `datos_desercion.csv` en la raíz del proyecto.
2. Asegúrate de que tenga:
   - Columna objetivo: `DESERTA` (1 = deserta, 0 = no deserta)
   - Columnas candidatas listadas en el script

3. Ejecuta:

```bash
python ga_seleccion_variables_v2.py
```

---

## Variables candidatas

Modifica la lista en el script:

```python
candidate_features = [
    "EDAD", "PROM_INGRESO", "PUNTAJE_PAA", "GENERO", "CASADO",
    "DIVORCIADO", "UNION_LIBRE", "TRABAJA", "DISCAPACIDAD"
]
```

---

## Salidas

Al terminar, se guardan:

- `ga_fitness_por_generacion.png`
- `ga_variables_seleccionadas.png`

---

## Notas metodológicas

- Fitness por defecto: **AUC promedio** con validación cruzada (cv=3).
- Si AUC falla por distribución/clases, usa **accuracy promedio**.
- El GA implementa:
  - torneo binario
  - cruce de un punto
  - mutación bit a bit
  - elitismo (1 individuo)

---

## ómo citar este repositorio

Si publicas artículos o tesis, puedes citarlo así:

> Flores Lira, A. F. (2025). *Selección de variables con Algoritmo Genético para predicción de deserción temprana* [Código fuente]. GitHub.  

Además incluye un archivo `CITATION.cff` listo para GitHub/Zenodo.

---

## Licencia

Este proyecto se publica bajo licencia **MIT**.  
Puedes usarlo, modificarlo y distribuirlo citando al autor.

---

## Agradecimientos

Se usan librerías open‑source ampliamente aceptadas en ML:

- NumPy
- Pandas
- Scikit‑Learn
- Matplotlib

---