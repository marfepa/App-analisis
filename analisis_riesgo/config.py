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

    # Columnas requeridas en CSV de calificaciones (flexibles - se intentará mapeo automático)
    'columnas_calificaciones': [
        'Nota',  # Solo esta es realmente requerida
    ],

    # Columnas opcionales en CSV de calificaciones
    'columnas_calificaciones_opcionales': [
        'IDEstudiante',
        'CursoID',
        'Evaluacion',
        'Competencia',
        'Fecha',
        'Peso',
        'Observaciones',
    ],

    # Mapeo de columnas alternativas para calificaciones
    'mapeo_columnas_calificaciones': {
        'IDEstudiante': ['IDEstudiante', 'id_estudiante', 'EstudianteID'],
        'NombreEstudiante': ['NombreEstudiante', 'nombre_estudiante', 'Nombre', 'nombre'],
        'CursoID': ['CursoID', 'curso_id', 'Curso', 'CursoEvaluado', 'curso'],
        'Evaluacion': ['Evaluacion', 'evaluacion', 'NombreInstrumento', 'Instrumento', 'instrumento'],
        'Nota': ['Nota', 'nota', 'PuntuacionCriterio', 'CalificacionTotalInstrumento', 'Calificacion', 'calificacion'],
        'Fecha': ['Fecha', 'fecha', 'FechaEvaluacion', 'fecha_evaluacion'],
        'Competencia': ['Competencia', 'competencia', 'NombreCriterioEvaluado', 'Criterio'],
        'Peso': ['Peso', 'peso', 'ponderacion', 'Ponderacion'],
    },

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
# CONFIGURACIÓN DE GUI (PyQt6)
# ============================================================================

GUI_CONFIG = {
    # Ventana principal
    'window': {
        'title': 'Análisis de Riesgo Académico - EF',
        'width': 800,
        'height': 600,
        'min_width': 600,
        'min_height': 400,
        'border_radius': 12,
    },

    # Colores principales (estilo Apple)
    'colors': {
        'background': '#F0F0F0',
        'background_opacity': 0.8,
        'glass_background': 'rgba(255, 255, 255, 0.8)',
        'glass_border': 'rgba(200, 200, 200, 0.3)',

        'primary': '#007AFF',           # Azul Apple
        'primary_hover': '#005BB5',
        'primary_disabled': '#B3D7FF',

        'text_primary': '#000000',
        'text_secondary': '#6E6E6E',
        'text_disabled': '#C7C7C7',

        'success': '#34C759',           # Verde Apple
        'warning': '#FF9500',           # Naranja Apple
        'error': '#FF3B30',             # Rojo Apple
        'info': '#5AC8FA',              # Azul claro Apple

        'border': '#D1D1D6',
        'shadow': 'rgba(0, 0, 0, 0.1)',
    },

    # Tipografía
    'fonts': {
        'family': 'Segoe UI, SF Pro Display, -apple-system, system-ui, sans-serif',
        'size_title': 24,
        'size_subtitle': 20,
        'size_heading': 18,
        'size_body': 14,
        'size_small': 12,
        'weight_regular': 400,
        'weight_medium': 500,
        'weight_bold': 600,
    },

    # Efectos visuales
    'effects': {
        'blur_radius': 10,
        'shadow_offset': (0, 2),
        'shadow_blur_radius': 8,
        'animation_duration': 300,      # ms
        'hover_scale': 1.02,
    },

    # Espaciado
    'spacing': {
        'xs': 4,
        'sm': 8,
        'md': 16,
        'lg': 24,
        'xl': 32,
        'xxl': 48,
    },

    # Componentes
    'header': {
        'height': 60,
        'padding': 20,
    },

    'button': {
        'height': 44,
        'padding_x': 24,
        'padding_y': 12,
        'border_radius': 8,
        'min_width': 100,
    },

    'input': {
        'height': 44,
        'padding_x': 16,
        'border_radius': 8,
    },

    'card': {
        'padding': 20,
        'border_radius': 12,
        'margin': 12,
    },

    'footer': {
        'height': 40,
        'padding': 12,
    },
}


# Estilos CSS para PyQt6
GUI_STYLES = {
    'main_window': """
        QMainWindow {
            background-color: #F0F0F0;
        }
    """,

    'glass_widget': """
        QWidget {
            background-color: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(200, 200, 200, 0.3);
            border-radius: 12px;
        }
    """,

    'primary_button': """
        QPushButton {
            background-color: #007AFF;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 14px;
            font-weight: 500;
            min-height: 44px;
        }
        QPushButton:hover {
            background-color: #005BB5;
        }
        QPushButton:pressed {
            background-color: #004494;
        }
        QPushButton:disabled {
            background-color: #B3D7FF;
            color: #FFFFFF;
        }
    """,

    'secondary_button': """
        QPushButton {
            background-color: rgba(255, 255, 255, 0.9);
            color: #007AFF;
            border: 1px solid #007AFF;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 14px;
            font-weight: 500;
            min-height: 44px;
        }
        QPushButton:hover {
            background-color: rgba(0, 122, 255, 0.1);
        }
        QPushButton:pressed {
            background-color: rgba(0, 122, 255, 0.2);
        }
    """,

    'upload_button': """
        QPushButton {
            background-color: rgba(255, 255, 255, 0.9);
            color: #000000;
            border: 2px dashed #D1D1D6;
            border-radius: 8px;
            padding: 32px;
            font-size: 14px;
            min-height: 120px;
        }
        QPushButton:hover {
            border-color: #007AFF;
            background-color: rgba(0, 122, 255, 0.05);
        }
    """,

    'header_label': """
        QLabel {
            color: #000000;
            font-size: 20px;
            font-weight: 600;
            background: transparent;
        }
    """,

    'body_label': """
        QLabel {
            color: #000000;
            font-size: 14px;
            background: transparent;
        }
    """,

    'secondary_label': """
        QLabel {
            color: #6E6E6E;
            font-size: 12px;
            background: transparent;
        }
    """,

    'line_edit': """
        QLineEdit {
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #D1D1D6;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 14px;
            min-height: 44px;
        }
        QLineEdit:focus {
            border-color: #007AFF;
        }
    """,

    'slider': """
        QSlider::groove:horizontal {
            background: #D1D1D6;
            height: 4px;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #007AFF;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #005BB5;
        }
    """,

    'progress_bar': """
        QProgressBar {
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #D1D1D6;
            border-radius: 8px;
            text-align: center;
            height: 24px;
        }
        QProgressBar::chunk {
            background-color: #007AFF;
            border-radius: 7px;
        }
    """,

    'table': """
        QTableWidget {
            background-color: rgba(255, 255, 255, 0.9);
            border: 1px solid #D1D1D6;
            border-radius: 8px;
            gridline-color: #E5E5E5;
            font-size: 14px;
        }
        QTableWidget::item {
            padding: 8px;
        }
        QHeaderView::section {
            background-color: rgba(240, 240, 240, 0.95);
            color: #000000;
            font-weight: 600;
            padding: 12px;
            border: none;
            border-bottom: 1px solid #D1D1D6;
        }
    """,

    'tab_widget': """
        QTabWidget::pane {
            border: 1px solid #D1D1D6;
            border-radius: 8px;
            background-color: rgba(255, 255, 255, 0.9);
        }
        QTabBar::tab {
            background-color: rgba(255, 255, 255, 0.6);
            color: #6E6E6E;
            padding: 12px 24px;
            border: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            margin-right: 4px;
        }
        QTabBar::tab:selected {
            background-color: rgba(255, 255, 255, 0.9);
            color: #007AFF;
            font-weight: 600;
        }
        QTabBar::tab:hover {
            background-color: rgba(0, 122, 255, 0.1);
        }
    """,

    'scroll_area': """
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QScrollBar:vertical {
            background-color: transparent;
            width: 8px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: rgba(0, 0, 0, 0.3);
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """,
}


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
    'GUI_CONFIG',
    'GUI_STYLES',
    'cargar_umbrales_personalizados',
    'validar_umbrales',
    'obtener_nivel_riesgo_asistencia',
    'obtener_nivel_riesgo_nota',
    'obtener_nivel_riesgo_combinado',
]
