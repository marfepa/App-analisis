# Claude.md: Guía para Desarrollar y Estructura del Proyecto de Análisis de Riesgo Académico

Este documento explica cómo usar Claude para desarrollar el programa de análisis de riesgo académico con machine learning, basado en la guía proporcionada. El proyecto convierte la skill descrita en un programa Python independiente que procesa CSVs de asistencia y calificaciones, aplica análisis descriptivo y ML, y genera reportes en Excel y Word.

## Propósito del Proyecto
- Analizar datos de Educación Física para identificar estudiantes en riesgo académico.
- Incorporar ML (e.g., clustering con KMeans) para predecir y clasificar riesgos.
- Generar reportes profesionales cumpliendo LOMLOE y RGPD (datos anonimizados).
- Ejecución: Script de consola con argumentos (e.g., `python main.py --asistencia file.csv --calificaciones file.csv`).

## Estructura del Proyecto
El proyecto está organizado en módulos para facilitar el desarrollo y mantenimiento. Cada archivo tiene un rol específico:
analisis_riesgo/
├── main.py                # Punto de entrada principal. Maneja argumentos CLI con argparse y orquesta el flujo: carga datos → análisis → generación de reportes.
├── data_loader.py         # Módulo para cargar y validar los CSVs. Incluye validación de columnas, manejo de fechas, encodings, y anonimización de datos.
├── analysis.py            # Lógica de análisis: descriptivo (agregados por curso, individuales) y ML (features, clustering con KMeans o regresión para predicción de riesgo).
├── report_generator.py    # Genera los archivos Excel (.xlsx) y Word (.docx) con estructuras definidas (hojas, tablas, gráficos, formatos condicionales).
├── config.py              # Configuraciones globales: umbrales por defecto como diccionario, y funciones para overrides via CLI.
├── outputs/               # Carpeta para guardar los reportes generados (e.g., analisis-riesgo-YYYYMMDD.xlsx).
├── requirements.txt       # Lista de dependencias: pandas, numpy, scikit-learn, openpyxl, python-docx.
├── tests/                 # (Opcional) Carpeta para tests con pytest (e.g., test_data_loader.py).
└── README.md              # Instrucciones de uso, instalación y ejecución.
text### Descripción Detallada de Archivos
- **config.py**: Define `UMBRALES` como dict. Incluye funciones para cargar overrides (e.g., desde JSON o args).
- **data_loader.py**:
  - Función principal: `load_data(asistencia_path, calificaciones_path, umbrales)`.
  - Valida estructuras, maneja errores comunes (delimitadores, encodings), convierte fechas, imputa valores faltantes.
  - Retorna DataFrames limpios.
- **analysis.py**:
  - Análisis descriptivo: Groupby para agregados, cálculos de tasas, patrones temporales.
  - ML: Crea df_features por estudiante, normaliza, aplica KMeans (n_clusters=3), asigna riesgos (ALTO/MEDIO/BAJO).
  - Opcional: Regresión para predecir notas finales.
  - Retorna dict con resultados.
- **report_generator.py**:
  - Funciones para Excel: Crea workbook, añade hojas, tablas, formatos condicionales, gráficos (usando openpyxl.chart).
  - Funciones para Word: Crea doc, añade secciones, tablas, párrafos con python-docx.
  - Incluye sección para insights de ML.
- **main.py**:
  - Parsea args (e.g., paths a CSVs, umbrales custom).
  - Ejecuta flujo completo.
  - Imprime resumen en consola y paths a outputs.
- **requirements.txt**:
pandas
numpy
scikit-learn
openpyxl
python-docx
text## Cómo Desarrollar con Claude
Usa Claude para generar código módulo por módulo. Ejemplo de prompts:
1. "Genera config.py completo basado en la guía."
2. "Ahora data_loader.py: Incluye validación de columnas requeridas y manejo de fechas."
3. "Para analysis.py: Implementa el análisis descriptivo y el ML con KMeans."
4. "report_generator.py: Código para generar Excel con 6 hojas y formatos."
5. "Finalmente, main.py y un README.md simple."

Itera: Ejecuta el código localmente, envía errores a Claude para fixes.

## Instalación y Ejecución
- Instala dependencias: `pip install -r requirements.txt`.
- Ejecuta: `python main.py --asistencia asistencia.csv --calificaciones calificaciones.csv --umbrales "{'asistencia_riesgo':80}"`.
- Outputs: En `./outputs/`.

## Consideraciones
- **RGPD**: Siempre anonimizar; eliminar columnas con nombres reales.
- **ML**: Usa unsupervised learning inicialmente; si tienes datos etiquetados, ajusta a supervisado.
- **Testing**: Agrega pytest para validar (e.g., carga de datos falsos).
- **Versión**: Python 3.10+.

Si necesitas expansiones, consulta la guía original o itera con Claude.
