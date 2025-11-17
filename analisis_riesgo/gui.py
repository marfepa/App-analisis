"""
Interfaz gráfica moderna con PyQt6 para el análisis de riesgo académico.

Diseño estilo Apple con efectos liquid glass, drag & drop, y animaciones suaves.
Cumple con RGPD y proporciona feedback visual claro al usuario.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
    QTabWidget, QTableWidget, QTableWidgetItem, QScrollArea,
    QSlider, QSpinBox, QGroupBox, QGridLayout, QHeaderView,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve,
    QRect, QSize, QTimer, QUrl
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QIcon, QPixmap, QPainter,
    QLinearGradient, QDragEnterEvent, QDropEvent
)

# Imports de módulos del proyecto
import config
import data_loader
import analysis
import report_generator


# ============================================================================
# WORKER THREADS (para no bloquear la GUI)
# ============================================================================

class AnalysisWorker(QThread):
    """Thread para ejecutar el análisis sin bloquear la interfaz."""

    # Señales
    progress = pyqtSignal(int, str)  # (porcentaje, mensaje)
    finished = pyqtSignal(dict)       # Resultados del análisis
    error = pyqtSignal(str)           # Mensaje de error

    def __init__(self, asistencia_path: str, calificaciones_path: str,
                 umbrales: Dict[str, Any]):
        super().__init__()
        self.asistencia_path = asistencia_path
        self.calificaciones_path = calificaciones_path
        self.umbrales = umbrales

    def run(self):
        """Ejecuta el análisis completo."""
        try:
            # 1. Cargar datos
            self.progress.emit(10, "Cargando datos de asistencia...")
            df_asistencia = data_loader.cargar_asistencia(self.asistencia_path)

            self.progress.emit(20, "Cargando datos de calificaciones...")
            df_calificaciones = data_loader.cargar_calificaciones(
                self.calificaciones_path
            )

            # 2. Validar datos
            self.progress.emit(30, "Validando datos...")
            data_loader.validar_datos(df_asistencia, df_calificaciones)

            # 3. Analizar asistencia
            self.progress.emit(40, "Analizando asistencia...")
            resultados_asist = analysis.analizar_asistencia(
                df_asistencia,
                self.umbrales
            )

            # 4. Analizar rendimiento
            self.progress.emit(60, "Analizando rendimiento académico...")
            resultados_rend = analysis.analizar_rendimiento(
                df_calificaciones,
                self.umbrales
            )

            # 5. Clasificar estudiantes
            self.progress.emit(70, "Clasificando estudiantes por riesgo...")
            clasificacion = analysis.clasificar_estudiantes(
                resultados_asist,
                resultados_rend,
                self.umbrales
            )

            # 6. Análisis ML
            self.progress.emit(80, "Ejecutando análisis con Machine Learning...")
            ml_results = analysis.analizar_con_ml(
                df_asistencia,
                df_calificaciones,
                self.umbrales
            )

            # 7. Compilar resultados
            self.progress.emit(95, "Compilando resultados...")
            resultados = {
                'asistencia': resultados_asist,
                'rendimiento': resultados_rend,
                'clasificacion': clasificacion,
                'ml': ml_results,
                'df_asistencia': df_asistencia,
                'df_calificaciones': df_calificaciones,
            }

            self.progress.emit(100, "Análisis completado")
            self.finished.emit(resultados)

        except Exception as e:
            self.error.emit(f"Error durante el análisis: {str(e)}")


class ReportWorker(QThread):
    """Thread para generar reportes sin bloquear la interfaz."""

    finished = pyqtSignal(str)  # Path del archivo generado
    error = pyqtSignal(str)

    def __init__(self, resultados: Dict, formato: str, output_dir: str):
        super().__init__()
        self.resultados = resultados
        self.formato = formato
        self.output_dir = output_dir

    def run(self):
        """Genera el reporte."""
        try:
            if self.formato == 'excel':
                output_path = report_generator.generar_reporte_excel(
                    self.resultados['asistencia'],
                    self.resultados['rendimiento'],
                    self.resultados['clasificacion'],
                    self.resultados['ml'],
                    output_dir=self.output_dir
                )
            elif self.formato == 'word':
                output_path = report_generator.generar_reporte_word(
                    self.resultados['asistencia'],
                    self.resultados['rendimiento'],
                    self.resultados['clasificacion'],
                    self.resultados['ml'],
                    output_dir=self.output_dir
                )
            else:
                raise ValueError(f"Formato desconocido: {self.formato}")

            self.finished.emit(output_path)

        except Exception as e:
            self.error.emit(f"Error al generar reporte: {str(e)}")


# ============================================================================
# WIDGETS PERSONALIZADOS
# ============================================================================

class GlassWidget(QWidget):
    """Widget con efecto liquid glass (vidrio esmerilado)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(config.GUI_STYLES['glass_widget'])


class UploadButton(QPushButton):
    """Botón de carga con drag & drop habilitado."""

    file_selected = pyqtSignal(str)  # Path del archivo

    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        self.label_text = label_text
        self.file_path = None

        # Configurar apariencia
        self.setStyleSheet(config.GUI_STYLES['upload_button'])
        self.setAcceptDrops(True)
        self.update_text()

        # Conectar click
        self.clicked.connect(self.select_file)

    def update_text(self):
        """Actualiza el texto del botón."""
        if self.file_path:
            filename = Path(self.file_path).name
            self.setText(f"📄 {filename}")
        else:
            self.setText(f"📤 {self.label_text}\n\n(Click o arrastra archivo)")

    def select_file(self):
        """Abre diálogo de selección de archivo."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Seleccionar {self.label_text}",
            "",
            "Archivos CSV (*.csv);;Todos los archivos (*.*)"
        )

        if file_path:
            self.set_file(file_path)

    def set_file(self, file_path: str):
        """Establece el archivo seleccionado."""
        self.file_path = file_path
        self.update_text()
        self.file_selected.emit(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Maneja el evento de arrastrar archivo sobre el botón."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                config.GUI_STYLES['upload_button'].replace(
                    'rgba(0, 122, 255, 0.05)',
                    'rgba(0, 122, 255, 0.15)'
                )
            )

    def dragLeaveEvent(self, event):
        """Restaura estilo al salir del área de drop."""
        self.setStyleSheet(config.GUI_STYLES['upload_button'])

    def dropEvent(self, event: QDropEvent):
        """Maneja el evento de soltar archivo."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.endswith('.csv'):
                self.set_file(file_path)
            else:
                QMessageBox.warning(
                    self,
                    "Archivo inválido",
                    "Por favor selecciona un archivo CSV."
                )

        self.setStyleSheet(config.GUI_STYLES['upload_button'])
        event.acceptProposedAction()


class ThresholdSettings(QGroupBox):
    """Panel de configuración de umbrales con sliders."""

    thresholds_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__("⚙️ Configuración de Umbrales", parent)
        self.umbrales = config.UMBRALES.copy()
        self.setup_ui()

    def setup_ui(self):
        """Configura la interfaz."""
        layout = QGridLayout()
        layout.setSpacing(config.GUI_CONFIG['spacing']['md'])

        # Umbrales de asistencia
        row = 0
        self.add_slider(
            layout, row,
            "Asistencia - Riesgo Alto (%):",
            'asistencia_riesgo',
            0, 100, 75
        )

        row += 1
        self.add_slider(
            layout, row,
            "Asistencia - Alerta (%):",
            'asistencia_alerta',
            0, 100, 85
        )

        # Umbrales de notas
        row += 1
        self.add_slider(
            layout, row,
            "Nota - Riesgo Alto (0-10):",
            'nota_riesgo',
            0, 10, 4
        )

        row += 1
        self.add_slider(
            layout, row,
            "Nota - Alerta (0-10):",
            'nota_alerta',
            0, 10, 6
        )

        # Botón restaurar defaults
        row += 1
        restore_btn = QPushButton("🔄 Restaurar Valores por Defecto")
        restore_btn.setStyleSheet(config.GUI_STYLES['secondary_button'])
        restore_btn.clicked.connect(self.restore_defaults)
        layout.addWidget(restore_btn, row, 0, 1, 3)

        self.setLayout(layout)
        self.setStyleSheet(config.GUI_STYLES['glass_widget'])

    def add_slider(self, layout: QGridLayout, row: int, label: str,
                   key: str, min_val: int, max_val: int, default: int):
        """Agrega un slider con label y valor."""
        # Label
        lbl = QLabel(label)
        lbl.setStyleSheet(config.GUI_STYLES['body_label'])
        layout.addWidget(lbl, row, 0)

        # Slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(int(self.umbrales.get(key, default)))
        slider.setStyleSheet(config.GUI_STYLES['slider'])
        slider.valueChanged.connect(lambda v: self.update_threshold(key, v))
        layout.addWidget(slider, row, 1)

        # Valor actual
        value_lbl = QLabel(str(slider.value()))
        value_lbl.setStyleSheet(config.GUI_STYLES['body_label'])
        value_lbl.setMinimumWidth(40)
        slider.valueChanged.connect(lambda v: value_lbl.setText(str(v)))
        layout.addWidget(value_lbl, row, 2)

        # Guardar referencias
        setattr(self, f'{key}_slider', slider)
        setattr(self, f'{key}_label', value_lbl)

    def update_threshold(self, key: str, value: int):
        """Actualiza un umbral y emite señal."""
        self.umbrales[key] = float(value)
        self.thresholds_changed.emit(self.umbrales)

    def restore_defaults(self):
        """Restaura umbrales por defecto."""
        self.umbrales = config.UMBRALES.copy()

        # Actualizar sliders
        for key in ['asistencia_riesgo', 'asistencia_alerta',
                    'nota_riesgo', 'nota_alerta']:
            slider = getattr(self, f'{key}_slider', None)
            if slider:
                slider.setValue(int(self.umbrales[key]))

        self.thresholds_changed.emit(self.umbrales)


# ============================================================================
# VENTANA PRINCIPAL
# ============================================================================

class MainWindow(QMainWindow):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()

        # Estado
        self.asistencia_path: Optional[str] = None
        self.calificaciones_path: Optional[str] = None
        self.resultados: Optional[Dict] = None
        self.umbrales = config.UMBRALES.copy()

        # Workers
        self.analysis_worker: Optional[AnalysisWorker] = None
        self.report_worker: Optional[ReportWorker] = None

        # Configurar ventana
        self.setup_window()
        self.setup_ui()

    def setup_window(self):
        """Configura la ventana principal."""
        cfg = config.GUI_CONFIG['window']

        self.setWindowTitle(cfg['title'])
        self.resize(cfg['width'], cfg['height'])
        self.setMinimumSize(cfg['min_width'], cfg['min_height'])

        # Estilo
        self.setStyleSheet(config.GUI_STYLES['main_window'])

    def setup_ui(self):
        """Configura la interfaz de usuario."""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(config.GUI_CONFIG['spacing']['lg'])

        # Header
        main_layout.addWidget(self.create_header())

        # Sección de carga de archivos
        main_layout.addWidget(self.create_upload_section())

        # Sección de configuración (colapsable)
        self.config_section = self.create_config_section()
        self.config_section.setVisible(False)
        main_layout.addWidget(self.config_section)

        # Botón principal de análisis
        main_layout.addWidget(self.create_analyze_button())

        # Progress bar (inicialmente oculto)
        self.progress_widget = self.create_progress_section()
        self.progress_widget.setVisible(False)
        main_layout.addWidget(self.progress_widget)

        # Sección de resultados (inicialmente oculta)
        self.results_widget = self.create_results_section()
        self.results_widget.setVisible(False)
        main_layout.addWidget(self.results_widget)

        # Footer
        main_layout.addWidget(self.create_footer())

    def create_header(self) -> QWidget:
        """Crea el header de la aplicación."""
        widget = GlassWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(20, 15, 20, 15)

        # Título
        title = QLabel("📊 Análisis de Riesgo Académico - EF")
        title.setStyleSheet(config.GUI_STYLES['header_label'])
        layout.addWidget(title)

        layout.addStretch()

        # Botón de configuración
        config_btn = QPushButton("⚙️")
        config_btn.setStyleSheet(config.GUI_STYLES['secondary_button'])
        config_btn.setFixedSize(44, 44)
        config_btn.setToolTip("Configurar umbrales")
        config_btn.clicked.connect(self.toggle_config)
        layout.addWidget(config_btn)

        return widget

    def create_upload_section(self) -> QWidget:
        """Crea la sección de carga de archivos."""
        widget = GlassWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)

        # Título
        title = QLabel("📁 Cargar Archivos CSV")
        title.setStyleSheet(config.GUI_STYLES['header_label'])
        layout.addWidget(title)

        # Advertencia RGPD
        rgpd_label = QLabel(
            "⚠️ Asegúrate de que los archivos usen IDs anonimizados (RGPD)"
        )
        rgpd_label.setStyleSheet(config.GUI_STYLES['secondary_label'])
        rgpd_label.setWordWrap(True)
        layout.addWidget(rgpd_label)

        # Botones de carga
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(config.GUI_CONFIG['spacing']['lg'])

        # Botón asistencia
        self.asistencia_btn = UploadButton("Asistencia (CSV)")
        self.asistencia_btn.file_selected.connect(self.on_asistencia_selected)
        buttons_layout.addWidget(self.asistencia_btn)

        # Botón calificaciones
        self.calificaciones_btn = UploadButton("Calificaciones (CSV)")
        self.calificaciones_btn.file_selected.connect(
            self.on_calificaciones_selected
        )
        buttons_layout.addWidget(self.calificaciones_btn)

        layout.addLayout(buttons_layout)

        # Labels de estado
        self.asistencia_status = QLabel("Sin archivo")
        self.asistencia_status.setStyleSheet(config.GUI_STYLES['secondary_label'])
        layout.addWidget(self.asistencia_status)

        self.calificaciones_status = QLabel("Sin archivo")
        self.calificaciones_status.setStyleSheet(
            config.GUI_STYLES['secondary_label']
        )
        layout.addWidget(self.calificaciones_status)

        return widget

    def create_config_section(self) -> QWidget:
        """Crea la sección de configuración."""
        self.threshold_settings = ThresholdSettings()
        self.threshold_settings.thresholds_changed.connect(
            self.on_thresholds_changed
        )
        return self.threshold_settings

    def create_analyze_button(self) -> QWidget:
        """Crea el botón principal de análisis."""
        self.analyze_btn = QPushButton("🔍 Analizar Datos")
        self.analyze_btn.setStyleSheet(config.GUI_STYLES['primary_button'])
        self.analyze_btn.setFixedHeight(60)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.start_analysis)

        return self.analyze_btn

    def create_progress_section(self) -> QWidget:
        """Crea la sección de progreso."""
        widget = GlassWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(config.GUI_STYLES['progress_bar'])
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Label de estado
        self.progress_label = QLabel("Analizando...")
        self.progress_label.setStyleSheet(config.GUI_STYLES['body_label'])
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        return widget

    def create_results_section(self) -> QWidget:
        """Crea la sección de resultados."""
        widget = GlassWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)

        # Título
        title = QLabel("📈 Resultados del Análisis")
        title.setStyleSheet(config.GUI_STYLES['header_label'])
        layout.addWidget(title)

        # Tabs
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet(config.GUI_STYLES['tab_widget'])

        # Tab 1: Resumen
        self.summary_tab = self.create_summary_tab()
        self.results_tabs.addTab(self.summary_tab, "📊 Resumen")

        # Tab 2: Listados
        self.list_tab = self.create_list_tab()
        self.results_tabs.addTab(self.list_tab, "📋 Listados")

        # Tab 3: Machine Learning
        self.ml_tab = self.create_ml_tab()
        self.results_tabs.addTab(self.ml_tab, "🤖 Machine Learning")

        layout.addWidget(self.results_tabs)

        # Botones de exportación
        export_layout = QHBoxLayout()
        export_layout.setSpacing(config.GUI_CONFIG['spacing']['md'])

        excel_btn = QPushButton("📥 Descargar Excel")
        excel_btn.setStyleSheet(config.GUI_STYLES['primary_button'])
        excel_btn.clicked.connect(lambda: self.export_report('excel'))
        export_layout.addWidget(excel_btn)

        word_btn = QPushButton("📥 Descargar Word")
        word_btn.setStyleSheet(config.GUI_STYLES['secondary_button'])
        word_btn.clicked.connect(lambda: self.export_report('word'))
        export_layout.addWidget(word_btn)

        layout.addLayout(export_layout)

        return widget

    def create_summary_tab(self) -> QWidget:
        """Crea el tab de resumen."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(config.GUI_STYLES['scroll_area'])

        self.summary_content = QLabel("Ejecuta un análisis para ver resultados")
        self.summary_content.setStyleSheet(config.GUI_STYLES['body_label'])
        self.summary_content.setWordWrap(True)
        self.summary_content.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(self.summary_content)
        layout.addWidget(scroll)

        return widget

    def create_list_tab(self) -> QWidget:
        """Crea el tab de listados."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tabla
        self.results_table = QTableWidget()
        self.results_table.setStyleSheet(config.GUI_STYLES['table'])
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            'ID Estudiante', 'Curso', 'Nivel de Riesgo', 'Detalles'
        ])

        # Ajustar columnas
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.results_table)

        return widget

    def create_ml_tab(self) -> QWidget:
        """Crea el tab de Machine Learning."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.ml_content = QLabel(
            "Los resultados de Machine Learning se mostrarán aquí"
        )
        self.ml_content.setStyleSheet(config.GUI_STYLES['body_label'])
        self.ml_content.setWordWrap(True)
        self.ml_content.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(self.ml_content)

        return widget

    def create_footer(self) -> QWidget:
        """Crea el footer."""
        widget = GlassWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(20, 10, 20, 10)

        # Texto
        footer_text = QLabel("Desarrollado con ❤️ - Cumple RGPD")
        footer_text.setStyleSheet(config.GUI_STYLES['secondary_label'])
        layout.addWidget(footer_text)

        layout.addStretch()

        # Botón ayuda
        help_btn = QPushButton("❓ Ayuda")
        help_btn.setStyleSheet(config.GUI_STYLES['secondary_button'])
        help_btn.clicked.connect(self.show_help)
        layout.addWidget(help_btn)

        return widget

    # ========================================================================
    # SLOTS Y EVENTOS
    # ========================================================================

    def toggle_config(self):
        """Muestra/oculta la sección de configuración."""
        self.config_section.setVisible(not self.config_section.isVisible())

    def on_asistencia_selected(self, path: str):
        """Maneja la selección de archivo de asistencia."""
        self.asistencia_path = path
        self.asistencia_status.setText(f"✅ {Path(path).name}")
        self.check_ready_to_analyze()

    def on_calificaciones_selected(self, path: str):
        """Maneja la selección de archivo de calificaciones."""
        self.calificaciones_path = path
        self.calificaciones_status.setText(f"✅ {Path(path).name}")
        self.check_ready_to_analyze()

    def on_thresholds_changed(self, umbrales: Dict):
        """Maneja cambios en umbrales."""
        self.umbrales = umbrales

    def check_ready_to_analyze(self):
        """Verifica si está listo para analizar."""
        ready = (self.asistencia_path is not None and
                 self.calificaciones_path is not None)
        self.analyze_btn.setEnabled(ready)

    def start_analysis(self):
        """Inicia el análisis."""
        # Deshabilitar botones
        self.analyze_btn.setEnabled(False)
        self.asistencia_btn.setEnabled(False)
        self.calificaciones_btn.setEnabled(False)

        # Mostrar progress
        self.progress_widget.setVisible(True)
        self.results_widget.setVisible(False)
        self.progress_bar.setValue(0)

        # Crear y ejecutar worker
        self.analysis_worker = AnalysisWorker(
            self.asistencia_path,
            self.calificaciones_path,
            self.umbrales
        )

        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)

        self.analysis_worker.start()

    def on_analysis_progress(self, percent: int, message: str):
        """Actualiza el progreso del análisis."""
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def on_analysis_finished(self, resultados: Dict):
        """Maneja finalización del análisis."""
        self.resultados = resultados

        # Ocultar progress
        self.progress_widget.setVisible(False)

        # Mostrar resultados
        self.display_results()
        self.results_widget.setVisible(True)

        # Re-habilitar botones
        self.analyze_btn.setEnabled(True)
        self.asistencia_btn.setEnabled(True)
        self.calificaciones_btn.setEnabled(True)

    def on_analysis_error(self, error_msg: str):
        """Maneja errores en el análisis."""
        # Ocultar progress
        self.progress_widget.setVisible(False)

        # Mostrar error
        QMessageBox.critical(
            self,
            "Error en el Análisis",
            f"Ocurrió un error durante el análisis:\n\n{error_msg}"
        )

        # Re-habilitar botones
        self.analyze_btn.setEnabled(True)
        self.asistencia_btn.setEnabled(True)
        self.calificaciones_btn.setEnabled(True)

    def display_results(self):
        """Muestra los resultados del análisis."""
        if not self.resultados:
            return

        # Tab 1: Resumen
        self.display_summary()

        # Tab 2: Listados
        self.display_lists()

        # Tab 3: ML
        self.display_ml_results()

    def display_summary(self):
        """Muestra el resumen."""
        clasificacion = self.resultados.get('clasificacion', {})

        # Contar por nivel de riesgo
        alto = len(clasificacion.get('ALTO', []))
        medio = len(clasificacion.get('MEDIO', []))
        alerta = len(clasificacion.get('ALERTA', []))
        optimo = len(clasificacion.get('OPTIMO', []))
        total = alto + medio + alerta + optimo

        summary_html = f"""
        <div style='padding: 20px;'>
            <h2>Resumen General</h2>
            <p><strong>Total de estudiantes analizados:</strong> {total}</p>
            <hr>
            <h3>Distribución por Nivel de Riesgo:</h3>
            <p style='color: #FF3B30; font-size: 16px;'>
                🔴 <strong>Riesgo Alto:</strong> {alto} estudiantes
                ({alto/total*100:.1f}% del total)
            </p>
            <p style='color: #FF9500; font-size: 16px;'>
                🟠 <strong>Riesgo Medio:</strong> {medio} estudiantes
                ({medio/total*100:.1f}% del total)
            </p>
            <p style='color: #FFCC00; font-size: 16px;'>
                🟡 <strong>Alerta:</strong> {alerta} estudiantes
                ({alerta/total*100:.1f}% del total)
            </p>
            <p style='color: #34C759; font-size: 16px;'>
                🟢 <strong>Óptimo:</strong> {optimo} estudiantes
                ({optimo/total*100:.1f}% del total)
            </p>
            <hr>
            <h3>Recomendaciones:</h3>
            <ul>
                <li>Priorizar intervención para estudiantes en riesgo alto</li>
                <li>Establecer seguimiento para estudiantes en riesgo medio</li>
                <li>Monitorizar estudiantes en alerta</li>
            </ul>
        </div>
        """

        self.summary_content.setText(summary_html)

    def display_lists(self):
        """Muestra los listados en la tabla."""
        clasificacion = self.resultados.get('clasificacion', {})

        # Limpiar tabla
        self.results_table.setRowCount(0)

        # Agregar filas
        row = 0
        for nivel in ['ALTO', 'MEDIO', 'ALERTA', 'OPTIMO']:
            estudiantes = clasificacion.get(nivel, [])

            for est_data in estudiantes:
                self.results_table.insertRow(row)

                # ID
                id_item = QTableWidgetItem(str(est_data.get('IDEstudiante', '')))
                self.results_table.setItem(row, 0, id_item)

                # Curso
                curso_item = QTableWidgetItem(str(est_data.get('CursoID', '')))
                self.results_table.setItem(row, 1, curso_item)

                # Nivel de riesgo
                nivel_item = QTableWidgetItem(nivel)

                # Color según nivel
                if nivel == 'ALTO':
                    nivel_item.setBackground(QColor(255, 59, 48, 50))
                elif nivel == 'MEDIO':
                    nivel_item.setBackground(QColor(255, 149, 0, 50))
                elif nivel == 'ALERTA':
                    nivel_item.setBackground(QColor(255, 204, 0, 50))
                else:
                    nivel_item.setBackground(QColor(52, 199, 89, 50))

                self.results_table.setItem(row, 2, nivel_item)

                # Detalles
                asist = est_data.get('PorcentajeAsistencia', 0)
                nota = est_data.get('NotaMedia', 0)
                detalles = f"Asistencia: {asist:.1f}% | Nota: {nota:.2f}"
                detalles_item = QTableWidgetItem(detalles)
                self.results_table.setItem(row, 3, detalles_item)

                row += 1

    def display_ml_results(self):
        """Muestra resultados de ML."""
        ml_results = self.resultados.get('ml', {})

        if not ml_results:
            self.ml_content.setText(
                "No hay suficientes datos para análisis ML"
            )
            return

        # Extraer información
        clusters = ml_results.get('clusters', {})
        predicciones = ml_results.get('predicciones', {})

        ml_html = f"""
        <div style='padding: 20px;'>
            <h2>Análisis con Machine Learning</h2>
            <h3>Clustering (K-Means)</h3>
            <p>Se identificaron {len(set(clusters.values()))} grupos de estudiantes
            con patrones similares.</p>

            <h3>Predicciones de Riesgo</h3>
            <p>El modelo ha clasificado automáticamente a los estudiantes
            según su probabilidad de riesgo académico.</p>

            <p><em>Nota: Para gráficos detallados, descarga el reporte Excel.</em></p>
        </div>
        """

        self.ml_content.setText(ml_html)

    def export_report(self, formato: str):
        """Exporta el reporte."""
        if not self.resultados:
            QMessageBox.warning(
                self,
                "Sin Resultados",
                "Primero debes ejecutar un análisis."
            )
            return

        # Seleccionar directorio de salida
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta de salida",
            str(Path.home() / "Downloads")
        )

        if not output_dir:
            return

        # Crear y ejecutar worker
        self.report_worker = ReportWorker(
            self.resultados,
            formato,
            output_dir
        )

        self.report_worker.finished.connect(self.on_report_finished)
        self.report_worker.error.connect(self.on_report_error)

        # Mostrar mensaje
        self.progress_label.setText(f"Generando reporte {formato.upper()}...")
        self.progress_widget.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Modo indeterminado

        self.report_worker.start()

    def on_report_finished(self, output_path: str):
        """Maneja finalización de generación de reporte."""
        self.progress_widget.setVisible(False)
        self.progress_bar.setRange(0, 100)

        QMessageBox.information(
            self,
            "Reporte Generado",
            f"Reporte guardado en:\n{output_path}"
        )

    def on_report_error(self, error_msg: str):
        """Maneja errores en generación de reporte."""
        self.progress_widget.setVisible(False)
        self.progress_bar.setRange(0, 100)

        QMessageBox.critical(
            self,
            "Error al Generar Reporte",
            f"Ocurrió un error:\n\n{error_msg}"
        )

    def show_help(self):
        """Muestra ayuda."""
        help_text = """
        <h2>Ayuda - Análisis de Riesgo Académico</h2>

        <h3>Paso 1: Cargar Archivos</h3>
        <p>Carga los archivos CSV de asistencia y calificaciones.
        Puedes hacer click o arrastrar los archivos.</p>

        <h3>Paso 2: Configurar (Opcional)</h3>
        <p>Haz click en ⚙️ para ajustar los umbrales de análisis.</p>

        <h3>Paso 3: Analizar</h3>
        <p>Haz click en "Analizar Datos" y espera los resultados.</p>

        <h3>Paso 4: Exportar</h3>
        <p>Descarga los reportes en Excel o Word.</p>

        <hr>
        <p><strong>Requisitos de los CSV:</strong></p>
        <ul>
            <li>Asistencia: IDEstudiante, CursoID, Fecha, Presente</li>
            <li>Calificaciones: IDEstudiante, CursoID, Evaluacion, Nota</li>
        </ul>

        <p><strong>IMPORTANTE:</strong> Los archivos deben usar IDs
        anonimizados para cumplir con RGPD.</p>
        """

        QMessageBox.information(self, "Ayuda", help_text)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def run_app():
    """Ejecuta la aplicación GUI."""
    app = QApplication(sys.argv)

    # Configurar fuente de la aplicación
    font = QFont(config.GUI_CONFIG['fonts']['family'])
    font.setPixelSize(config.GUI_CONFIG['fonts']['size_body'])
    app.setFont(font)

    # Crear y mostrar ventana
    window = MainWindow()
    window.show()

    # Ejecutar
    sys.exit(app.exec())


if __name__ == '__main__':
    run_app()
