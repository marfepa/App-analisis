# 📊 Análisis de Riesgo Académico con Machine Learning

Sistema completo de análisis de riesgo académico para Educación Física que incorpora **Machine Learning** para identificar automáticamente patrones de riesgo y generar reportes detallados en Excel y Word.

## 🎯 Características Principales

- **Análisis Descriptivo Completo**: Métricas de asistencia, rendimiento y competencias
- **Machine Learning**: Clustering automático (KMeans) para identificar grupos de riesgo
- **Reportes Profesionales**: Excel con tablas, gráficos y formatos; Word con resúmenes ejecutivos
- **Cumplimiento RGPD**: Anonimización automática de datos sensibles
- **Altamente Configurable**: Umbrales personalizables, filtros por curso, múltiples formatos
- **Robusto**: Manejo inteligente de errores, múltiples encodings y delimitadores CSV

## 📋 Requisitos

- **Python**: 3.10 o superior
- **Sistema Operativo**: Linux, macOS, Windows
- **Hardware**: CPU estándar (no requiere GPU)

## 🚀 Instalación

### 1. Clonar o descargar el proyecto

```bash
git clone <url-del-repositorio>
cd App-analisis
```

### 2. Crear entorno virtual (recomendado)

```bash
# Con virtualenv
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# O con conda
conda create -n riesgo-academico python=3.10
conda activate riesgo-academico
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 📂 Estructura del Proyecto

```
App-analisis/
├── analisis_riesgo/          # Código fuente principal
│   ├── main.py               # Punto de entrada del programa
│   ├── config.py             # Configuración y umbrales
│   ├── data_loader.py        # Carga y validación de datos
│   ├── analysis.py           # Análisis descriptivo y ML
│   └── report_generator.py   # Generación de reportes
├── outputs/                  # Reportes generados (se crea automáticamente)
├── requirements.txt          # Dependencias Python
└── README.md                 # Este archivo
```

## 📊 Formato de Datos de Entrada

### Archivo de Asistencia (CSV)

**Columnas requeridas:**
- `IDEstudiante`: ID anonimizado del estudiante (ej: EST001)
- `CursoID`: Identificador del curso (ej: 3ESO-A)
- `Fecha`: Fecha de la sesión (formato: YYYY-MM-DD o DD/MM/YYYY)
- `Presente`: Booleano indicando si asistió (true/false, 1/0, sí/no)

**Columnas opcionales:**
- `Retraso`: Booleano indicando si llegó tarde
- `FaltaJustificada`: Booleano indicando si la falta está justificada
- `Observaciones`: Notas adicionales

**Ejemplo:**
```csv
IDEstudiante,CursoID,Fecha,Presente,Retraso,FaltaJustificada
EST001,3ESO-A,2024-01-10,true,false,false
EST001,3ESO-A,2024-01-12,false,false,true
EST002,3ESO-A,2024-01-10,true,true,false
```

### Archivo de Calificaciones (CSV)

**Columnas requeridas:**
- `IDEstudiante`: ID anonimizado del estudiante
- `CursoID`: Identificador del curso
- `Evaluacion`: Nombre/ID de la evaluación (ej: Eval1, ExamenFinal)
- `Nota`: Calificación numérica (0-10)

**Columnas opcionales:**
- `Competencia`: Código de competencia (CE1, CE2, CE3, CE4, CE5)
- `Fecha`: Fecha de la evaluación
- `Peso`: Peso de la evaluación (default: 1.0)
- `Observaciones`: Notas adicionales

**Ejemplo:**
```csv
IDEstudiante,CursoID,Evaluacion,Nota,Competencia,Peso
EST001,3ESO-A,Eval1,7.5,CE1,1.0
EST001,3ESO-A,Eval1,8.0,CE2,1.0
EST002,3ESO-A,Eval1,6.5,CE1,1.0
```

## 💻 Uso

### Uso Básico

```bash
cd analisis_riesgo
python main.py --asistencia ../datos/asistencia.csv --calificaciones ../datos/calificaciones.csv
```

### Ejemplos Avanzados

#### 1. Con umbrales personalizados (JSON inline)

```bash
python main.py \
  --asistencia ../datos/asistencia.csv \
  --calificaciones ../datos/calificaciones.csv \
  --umbrales '{"asistencia_riesgo": 70, "nota_aprobado": 5.5}'
```

#### 2. Con umbrales desde archivo JSON

Crear archivo `umbrales_custom.json`:
```json
{
  "asistencia_riesgo": 70,
  "asistencia_alerta": 82,
  "nota_aprobado": 5.5,
  "nota_riesgo": 3.5
}
```

Ejecutar:
```bash
python main.py \
  --asistencia ../datos/asistencia.csv \
  --calificaciones ../datos/calificaciones.csv \
  --umbrales-file umbrales_custom.json
```

#### 3. Filtrar por curso específico

```bash
python main.py \
  --asistencia ../datos/asistencia.csv \
  --calificaciones ../datos/calificaciones.csv \
  --curso "3ESO-A"
```

#### 4. Generar solo Excel o solo Word

```bash
# Solo Excel
python main.py \
  --asistencia ../datos/asistencia.csv \
  --calificaciones ../datos/calificaciones.csv \
  --formato excel

# Solo Word
python main.py \
  --asistencia ../datos/asistencia.csv \
  --calificaciones ../datos/calificaciones.csv \
  --formato word
```

#### 5. Directorio de salida personalizado

```bash
python main.py \
  --asistencia ../datos/asistencia.csv \
  --calificaciones ../datos/calificaciones.csv \
  --output ../reportes/2024/
```

#### 6. Desactivar Machine Learning

```bash
python main.py \
  --asistencia ../datos/asistencia.csv \
  --calificaciones ../datos/calificaciones.csv \
  --no-ml
```

#### 7. Modo verbose (debugging)

```bash
python main.py \
  --asistencia ../datos/asistencia.csv \
  --calificaciones ../datos/calificaciones.csv \
  --verbose
```

### Ver ayuda completa

```bash
python main.py --help
```

## 🤖 Machine Learning: Cómo Funciona

### Ingeniería de Features

El sistema crea automáticamente un conjunto de features por estudiante:

**Asistencia:**
- Porcentaje de asistencia
- Número total de faltas (justificadas/injustificadas)
- Número de retrasos
- Ratios (faltas/sesiones, retrasos/sesiones)
- Tendencia temporal (mejorando/empeorando/estable)

**Rendimiento:**
- Nota media ponderada
- Desviación estándar de notas
- Número de evaluaciones
- Estado de aprobado/suspenso
- Tendencia temporal
- Notas por competencia (CE1-CE5)

**Combinadas:**
- Ratio rendimiento/asistencia
- Score de riesgo manual (basado en umbrales)

### Algoritmo: KMeans Clustering

1. **Normalización**: Los features se escalan con StandardScaler para que todos tengan la misma importancia
2. **Clustering**: KMeans agrupa a los estudiantes en clusters (por defecto: 3)
3. **Optimización**: El sistema busca automáticamente el número óptimo de clusters usando Silhouette Score
4. **Interpretación**: Los clusters se ordenan por nivel de riesgo y se asignan etiquetas (ALTO, MEDIO, ALERTA, ÓPTIMO)

### Métricas de Evaluación

- **Silhouette Score**: Mide la calidad del clustering (rango: -1 a 1, mayor es mejor)
- **Distribución de Clusters**: Se analiza la homogeneidad dentro de cada cluster

## 📈 Reportes Generados

### Reporte Excel (.xlsx)

**Hojas incluidas:**
1. **Resumen**: Vista general con distribución de riesgo
2. **Análisis por Curso**: Métricas agregadas por curso
3. **Listado Estudiantes**: Detalle individual con formato condicional (colores semáforo)
4. **Machine Learning**: Resultados de clustering y predicciones

**Características:**
- Tablas formateadas con estilos profesionales
- Formatos condicionales (colores según nivel de riesgo)
- Gráficos de distribución
- Fórmulas dinámicas

### Reporte Word (.docx)

**Secciones incluidas:**
1. **Resumen Ejecutivo**: Estadísticas generales y distribución de riesgo
2. **Análisis por Curso**: Detalle de cada curso
3. **Estudiantes en Riesgo**: Listados de estudiantes ALTO y MEDIO con métricas
4. **Machine Learning**: Explicación de clusters identificados
5. **Recomendaciones**: Sugerencias de acción basadas en los resultados

**Características:**
- Formato profesional con títulos y subtítulos
- Emojis para mejor visualización (🔴 ALTO, 🟠 MEDIO, 🟡 ALERTA, 🟢 ÓPTIMO)
- Tablas resumen
- Recomendaciones personalizadas

## ⚙️ Configuración Avanzada

### Umbrales Configurables

Todos los umbrales están definidos en `config.py` y pueden personalizarse:

```python
UMBRALES = {
    # Asistencia (%)
    'asistencia_riesgo': 75.0,      # < 75% = RIESGO ALTO
    'asistencia_alerta': 85.0,      # 75-85% = RIESGO MEDIO
    'asistencia_optima': 95.0,      # > 95% = ÓPTIMO

    # Notas (0-10)
    'nota_aprobado': 5.0,
    'nota_riesgo': 4.0,             # < 4 = RIESGO ALTO
    'nota_alerta': 6.0,             # 4-6 = RIESGO MEDIO
    'nota_excelente': 8.0,          # > 8 = EXCELENTE

    # Machine Learning
    'ml_n_clusters': 3,             # Número de clusters
    'ml_min_samples': 5,            # Mínimo de estudiantes para ML
    'ml_random_state': 42,          # Seed para reproducibilidad

    # ... más umbrales disponibles
}
```

### Competencias de Educación Física

El sistema soporta 5 competencias específicas (CE1-CE5):

- **CE1**: Resolución de problemas motrices
- **CE2**: Gestión de vida activa y saludable
- **CE3**: Interacción social y cooperación
- **CE4**: Expresión corporal y creatividad
- **CE5**: Valoración crítica y reflexión

## 🔒 Cumplimiento RGPD

El sistema está diseñado para cumplir con el RGPD:

- ✅ **Anonimización automática**: Elimina columnas con datos personales (Nombre, Email, etc.)
- ✅ **Solo IDs**: Procesa únicamente identificadores anonimizados
- ✅ **Advertencias**: Muestra avisos RGPD al inicio y en reportes
- ✅ **Sin almacenamiento**: No guarda datos personales, solo reportes agregados

**Importante**: Asegúrate de que los IDs en tus CSVs sean **anonimizados** antes de procesarlos.

## 🧪 Testing

### Ejecutar tests individuales

Cada módulo tiene una función de test incorporada:

```bash
cd analisis_riesgo

# Test data loader
python data_loader.py

# Test analysis
python analysis.py

# Test report generator
python report_generator.py
```

## 🐛 Solución de Problemas

### Error: "Archivo no encontrado"
- Verifica que las rutas sean correctas
- Usa rutas absolutas o relativas desde donde ejecutas el script

### Error: "Columnas faltantes"
- Revisa que tu CSV tenga todas las columnas requeridas
- Los nombres de columnas son case-sensitive

### Error: "No se puede leer el CSV"
- Prueba diferentes encodings: UTF-8, ISO-8859-1, Latin1
- Verifica el delimitador (`,` `;` o `\t`)
- El sistema auto-detecta, pero puedes forzar en `data_loader.py`

### Advertencia: "Pocas muestras para ML"
- El ML requiere mínimo 5 estudiantes (configurable)
- Para análisis pequeños, usa `--no-ml`

### Error: "Permisos denegados al guardar"
- Verifica permisos de escritura en el directorio de salida
- Cierra archivos Excel/Word abiertos antes de regenerar

## 📚 Dependencias

Principales librerías utilizadas:

- **pandas**: Manipulación y análisis de datos
- **numpy**: Cálculos numéricos
- **scikit-learn**: Machine Learning (KMeans, StandardScaler)
- **scipy**: Estadísticas avanzadas
- **openpyxl**: Generación de archivos Excel
- **python-docx**: Generación de documentos Word

Ver `requirements.txt` para versiones completas.

## 🤝 Contribuciones

Este es un proyecto educativo. Para mejoras:

1. Fork del repositorio
2. Crear rama con tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Add nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

MIT License - Libre uso para propósitos educativos y profesionales.

## 👨‍💻 Autor

Sistema de Análisis de Riesgo Académico
Versión 1.0.0

## 📞 Soporte

Para reportar bugs o solicitar features, abre un issue en el repositorio.

---

**¡Desarrollado con ❤️ para mejorar la educación!**
