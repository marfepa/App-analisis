"""
Configuración y umbrales para el análisis de riesgo académico.

Este módulo define todos los umbrales y parámetros configurables
utilizados en el análisis de riesgo académico en Educación Física.
Cumple con RGPD al no almacenar datos personales.
"""

import json
from typing import Dict, Any


# ============================================================================
# UMBRALES POR DEFECTO
# ============================================================================

UMBRALES = {
    # Umbrales de Asistencia (%)
    'asistencia_riesgo': 75.0,      # Debajo de este % = RIESGO ALTO
    'asistencia_alerta': 85.0,      # Entre este y riesgo = ALERTA
    'asistencia_optima': 95.0,      # Arriba de este % = ÓPTIMO

    # Umbrales de Rendimiento Académico (notas 0-10)
    'nota_aprobado': 5.0,           # Nota mínima para aprobar
    'nota_riesgo': 4.0,             # Debajo de este = RIESGO ALTO
    'nota_alerta': 6.0,             # Entre riesgo y este = ALERTA
    'nota_excelente': 8.0,          # Arriba de este = EXCELENTE

    # Umbrales de Competencias (niveles 1-4)
    'competencia_riesgo': 2.0,      # Nivel promedio < 2 = RIESGO
    'competencia_alerta': 2.5,      # Entre 2 y 2.5 = ALERTA
    'competencia_optima': 3.5,      # >= 3.5 = ÓPTIMO

    # Umbrales de Faltas y Retrasos (conteo absoluto)
    'max_faltas_justificadas': 3,   # Máximo aceptable de faltas justificadas
    'max_faltas_injustificadas': 1, # Máximo aceptable de faltas injustificadas
    'max_retrasos': 5,              # Máximo aceptable de retrasos

    # Umbrales de Evaluación
    'min_evaluaciones': 3,          # Mínimo de evaluaciones para análisis válido
    'min_sesiones': 10,             # Mínimo de sesiones para análisis válido

    # Umbrales de Machine Learning
    'ml_n_clusters': 3,             # Número de clusters para KMeans
    'ml_min_samples': 5,            # Mínimo de estudiantes para entrenar ML
    'ml_test_size': 0.2,            # Proporción para test set (si aplica)
    'ml_random_state': 42,          # Seed para reproducibilidad

    # Umbrales de Tendencias (cambio %)
    'tendencia_positiva': 5.0,      # Mejora > 5% = tendencia positiva
    'tendencia_negativa': -5.0,     # Caída > 5% = tendencia negativa
}


# ============================================================================
# NIVELES DE RIESGO
# ============================================================================

NIVELES_RIESGO = {
    'ALTO': {
        'codigo': 3,
        'color': 'FF0000',          # Rojo
        'emoji': '🔴',
        'descripcion': 'Riesgo Alto - Requiere intervención inmediata'
    },
    'MEDIO': {
        'codigo': 2,
        'color': 'FFA500',          # Naranja
        'emoji': '🟠',
        'descripcion': 'Riesgo Medio - Requiere seguimiento estrecho'
    },
    'ALERTA': {
        'codigo': 1,
        'color': 'FFFF00',          # Amarillo
        'emoji': '🟡',
        'descripcion': 'Alerta - Requiere monitorización'
    },
    'OPTIMO': {
        'codigo': 0,
        'color': '00FF00',          # Verde
        'emoji': '🟢',
        'descripcion': 'Óptimo - Sin intervención necesaria'
    }
}


# ============================================================================
# COMPETENCIAS ESPECÍFICAS DE EDUCACIÓN FÍSICA
# ============================================================================

COMPETENCIAS = {
    'CE1': {
        'nombre': 'Resolución de problemas motrices',
        'descripcion': 'Capacidad para resolver situaciones motrices variadas'
    },
    'CE2': {
        'nombre': 'Gestión de vida activa y saludable',
        'descripcion': 'Adopción de hábitos de vida saludable y actividad física'
    },
    'CE3': {
        'nombre': 'Interacción social y cooperación',
        'descripcion': 'Habilidades sociales y trabajo en equipo'
    },
    'CE4': {
        'nombre': 'Expresión corporal y creatividad',
        'descripcion': 'Expresión a través del movimiento y creatividad motriz'
    },
    'CE5': {
        'nombre': 'Valoración crítica y reflexión',
        'descripcion': 'Análisis crítico de la actividad física y deporte'
    }
}


# ============================================================================
# CONFIGURACIÓN DE REPORTES
# ============================================================================

REPORTE_CONFIG = {
    # Configuración de Excel
    'excel': {
        'usar_colores_semaforo': True,
        'incluir_graficos': True,
        'incluir_formulas': True,
        'incluir_tablas': True,
        'ancho_columna_default': 15,
        'alto_fila_header': 20,
    },

    # Configuración de Word
    'word': {
        'usar_emojis': True,
        'incluir_graficos': False,  # Gráficos se generan mejor en Excel
        'estilo_titulo': 'Heading 1',
        'estilo_subtitulo': 'Heading 2',
        'fuente': 'Calibri',
        'tamano_fuente': 11,
    },

    # Formato de fechas
    'formato_fecha': '%Y-%m-%d',
    'formato_fecha_reporte': '%d/%m/%Y',
    'formato_datetime': '%Y-%m-%d %H:%M:%S',
}


# ============================================================================
# CONFIGURACIÓN DE DATOS
# ============================================================================

DATOS_CONFIG = {
    # Columnas requeridas en CSV de asistencia
    'columnas_asistencia': [
        'IDEstudiante',
        'CursoID',
        'Fecha',
        'Presente',
    ],

    # Columnas opcionales en CSV de asistencia
    'columnas_asistencia_opcionales': [
        'Retraso',
        'FaltaJustificada',
        'Observaciones',
    ],

    # Columnas requeridas en CSV de calificaciones
    'columnas_calificaciones': [
        'IDEstudiante',
        'CursoID',
        'Evaluacion',
        'Nota',
    ],

    # Columnas opcionales en CSV de calificaciones
    'columnas_calificaciones_opcionales': [
        'Competencia',
        'Fecha',
        'Peso',
        'Observaciones',
    ],

    # Delimitadores CSV a probar
    'delimitadores_csv': [',', ';', '\t'],

    # Encodings a probar
    'encodings': ['utf-8', 'iso-8859-1', 'latin1', 'cp1252'],

    # Columnas a anonimizar/eliminar
    'columnas_sensibles': [
        'NombreEstudiante',
        'Nombre',
        'Apellido',
        'Email',
        'Telefono',
        'Direccion',
    ],
}


# ============================================================================
# MENSAJES Y TEXTOS
# ============================================================================

MENSAJES = {
    'advertencia_rgpd': (
        "⚠️  AVISO DE PROTECCIÓN DE DATOS (RGPD)\n"
        "Este análisis utiliza únicamente IDs anonimizados.\n"
        "NO procese nombres reales ni datos personales identificables.\n"
        "Asegúrese de que los datos de entrada cumplan con el RGPD.\n"
    ),

    'recomendaciones': {
        'ALTO_asistencia': (
            "🔴 URGENTE: Contactar a familias inmediatamente. "
            "Considerar plan de seguimiento personalizado."
        ),
        'ALTO_rendimiento': (
            "🔴 URGENTE: Revisar metodología y adaptar actividades. "
            "Plantear refuerzo individual o en pequeño grupo."
        ),
        'MEDIO_asistencia': (
            "🟠 IMPORTANTE: Comunicar con familias. "
            "Investigar causas y establecer plan de mejora."
        ),
        'MEDIO_rendimiento': (
            "🟠 IMPORTANTE: Revisar estrategias didácticas. "
            "Considerar adaptaciones curriculares."
        ),
        'ALERTA': (
            "🟡 Mantener seguimiento cercano. "
            "Prevenir deterioro mediante monitorización."
        ),
    },
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def cargar_umbrales_personalizados(json_string: str = None,
                                   json_file: str = None) -> Dict[str, Any]:
    """
    Carga umbrales personalizados desde JSON string o archivo.

    Args:
        json_string: String JSON con umbrales personalizados
        json_file: Path a archivo JSON con umbrales

    Returns:
        Dict con umbrales actualizados

    Raises:
        ValueError: Si el JSON es inválido
    """
    umbrales = UMBRALES.copy()

    try:
        if json_string:
            custom = json.loads(json_string)
            umbrales.update(custom)
        elif json_file:
            with open(json_file, 'r', encoding='utf-8') as f:
                custom = json.load(f)
            umbrales.update(custom)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error al parsear JSON de umbrales: {e}")
    except FileNotFoundError:
        raise ValueError(f"Archivo de umbrales no encontrado: {json_file}")

    return umbrales


def validar_umbrales(umbrales: Dict[str, Any]) -> bool:
    """
    Valida que los umbrales tengan valores coherentes.

    Args:
        umbrales: Diccionario de umbrales a validar

    Returns:
        True si son válidos

    Raises:
        ValueError: Si los umbrales son incoherentes
    """
    # Validar que riesgo < alerta < optima
    if umbrales['asistencia_riesgo'] >= umbrales['asistencia_alerta']:
        raise ValueError(
            "asistencia_riesgo debe ser menor que asistencia_alerta"
        )

    if umbrales['asistencia_alerta'] >= umbrales['asistencia_optima']:
        raise ValueError(
            "asistencia_alerta debe ser menor que asistencia_optima"
        )

    if umbrales['nota_riesgo'] >= umbrales['nota_alerta']:
        raise ValueError(
            "nota_riesgo debe ser menor que nota_alerta"
        )

    # Validar rangos válidos
    if not (0 <= umbrales['asistencia_riesgo'] <= 100):
        raise ValueError(
            "asistencia_riesgo debe estar entre 0 y 100"
        )

    if not (0 <= umbrales['nota_aprobado'] <= 10):
        raise ValueError(
            "nota_aprobado debe estar entre 0 y 10"
        )

    if umbrales['ml_n_clusters'] < 2:
        raise ValueError(
            "ml_n_clusters debe ser al menos 2"
        )

    return True


def obtener_nivel_riesgo_asistencia(porcentaje: float,
                                    umbrales: Dict[str, Any] = None) -> str:
    """
    Determina el nivel de riesgo basado en porcentaje de asistencia.

    Args:
        porcentaje: Porcentaje de asistencia (0-100)
        umbrales: Diccionario de umbrales (usa UMBRALES si es None)

    Returns:
        Nivel de riesgo: 'ALTO', 'MEDIO', 'ALERTA', 'OPTIMO'
    """
    if umbrales is None:
        umbrales = UMBRALES

    if porcentaje < umbrales['asistencia_riesgo']:
        return 'ALTO'
    elif porcentaje < umbrales['asistencia_alerta']:
        return 'MEDIO'
    elif porcentaje < umbrales['asistencia_optima']:
        return 'ALERTA'
    else:
        return 'OPTIMO'


def obtener_nivel_riesgo_nota(nota: float,
                              umbrales: Dict[str, Any] = None) -> str:
    """
    Determina el nivel de riesgo basado en nota.

    Args:
        nota: Nota (0-10)
        umbrales: Diccionario de umbrales (usa UMBRALES si es None)

    Returns:
        Nivel de riesgo: 'ALTO', 'MEDIO', 'ALERTA', 'OPTIMO'
    """
    if umbrales is None:
        umbrales = UMBRALES

    if nota < umbrales['nota_riesgo']:
        return 'ALTO'
    elif nota < umbrales['nota_alerta']:
        return 'MEDIO'
    elif nota < umbrales['nota_excelente']:
        return 'ALERTA'
    else:
        return 'OPTIMO'


def obtener_nivel_riesgo_combinado(nivel_asistencia: str,
                                   nivel_rendimiento: str) -> str:
    """
    Combina niveles de riesgo de asistencia y rendimiento.
    Usa el nivel más alto (más grave).

    Args:
        nivel_asistencia: Nivel de riesgo de asistencia
        nivel_rendimiento: Nivel de riesgo de rendimiento

    Returns:
        Nivel de riesgo combinado
    """
    orden = {'OPTIMO': 0, 'ALERTA': 1, 'MEDIO': 2, 'ALTO': 3}

    codigo_asist = orden.get(nivel_asistencia, 0)
    codigo_rend = orden.get(nivel_rendimiento, 0)

    codigo_max = max(codigo_asist, codigo_rend)

    for nivel, cod in orden.items():
        if cod == codigo_max:
            return nivel

    return 'OPTIMO'


# ============================================================================
# EXPORTAR CONFIGURACIÓN
# ============================================================================

__all__ = [
    'UMBRALES',
    'NIVELES_RIESGO',
    'COMPETENCIAS',
    'REPORTE_CONFIG',
    'DATOS_CONFIG',
    'MENSAJES',
    'cargar_umbrales_personalizados',
    'validar_umbrales',
    'obtener_nivel_riesgo_asistencia',
    'obtener_nivel_riesgo_nota',
    'obtener_nivel_riesgo_combinado',
]
