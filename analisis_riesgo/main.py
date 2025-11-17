#!/usr/bin/env python3
"""
Programa principal de análisis de riesgo académico con Machine Learning.

Este programa analiza datos de asistencia y calificaciones de estudiantes
de Educación Física, identifica patrones de riesgo académico utilizando
machine learning, y genera reportes detallados en Excel y Word.

Uso GUI (por defecto):
    python main.py

Uso CLI (con argumentos):
    python main.py --asistencia datos/asistencia.csv --calificaciones datos/calificaciones.csv

Autor: Sistema de Análisis de Riesgo Académico
Versión: 2.0.0
Licencia: MIT
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Importar módulos del proyecto
from config import (
    UMBRALES,
    MENSAJES,
    cargar_umbrales_personalizados,
    validar_umbrales,
)
from data_loader import cargar_datos
from analysis import realizar_analisis_completo
from report_generator import generar_reportes

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# BANNER Y PRESENTACIÓN
# ============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║     📊 ANÁLISIS DE RIESGO ACADÉMICO CON MACHINE LEARNING 📊             ║
║                     Educación Física                                    ║
║                                                                          ║
║     Version 1.0.0                                                       ║
║     Powered by Python + scikit-learn                                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """Imprime el banner de bienvenida."""
    print(BANNER)
    print(MENSAJES['advertencia_rgpd'])
    print()


# ============================================================================
# PARSEO DE ARGUMENTOS
# ============================================================================

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con los argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Análisis de Riesgo Académico con Machine Learning para Educación Física',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Análisis básico
  python main.py --asistencia datos/asistencia.csv --calificaciones datos/calificaciones.csv

  # Con umbrales personalizados
  python main.py --asistencia datos/asistencia.csv --calificaciones datos/calificaciones.csv \\
                 --umbrales '{"asistencia_riesgo": 70, "nota_aprobado": 5.5}'

  # Filtrar por curso específico
  python main.py --asistencia datos/asistencia.csv --calificaciones datos/calificaciones.csv \\
                 --curso "3ESO-A"

  # Generar solo Excel
  python main.py --asistencia datos/asistencia.csv --calificaciones datos/calificaciones.csv \\
                 --formato excel

  # Con directorio de salida personalizado
  python main.py --asistencia datos/asistencia.csv --calificaciones datos/calificaciones.csv \\
                 --output reportes/

Para más información, consulta el README.md
        """
    )

    # Argumentos de datos (opcionales para permitir modo GUI)
    parser.add_argument(
        '--asistencia',
        type=str,
        required=False,
        default=None,
        help='Ruta al archivo CSV con datos de asistencia (modo CLI)'
    )

    parser.add_argument(
        '--calificaciones',
        type=str,
        required=False,
        default=None,
        help='Ruta al archivo CSV con datos de calificaciones (modo CLI)'
    )

    # Modo GUI/CLI
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Forzar modo GUI (por defecto si no hay argumentos CLI)'
    )

    parser.add_argument(
        '--cli',
        action='store_true',
        help='Forzar modo CLI (requiere --asistencia y --calificaciones)'
    )

    # Argumentos opcionales
    parser.add_argument(
        '--umbrales',
        type=str,
        default=None,
        help='JSON string con umbrales personalizados (ej: \'{"asistencia_riesgo": 70}\')'
    )

    parser.add_argument(
        '--umbrales-file',
        type=str,
        default=None,
        help='Ruta a archivo JSON con umbrales personalizados'
    )

    parser.add_argument(
        '--curso',
        type=str,
        default=None,
        help='Filtrar análisis por un curso específico (ej: "3ESO-A")'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Directorio de salida para los reportes (default: outputs/)'
    )

    parser.add_argument(
        '--formato',
        type=str,
        choices=['excel', 'word', 'ambos'],
        default='ambos',
        help='Formato de reporte a generar (default: ambos)'
    )

    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Desactivar análisis con Machine Learning'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verbose: muestra información detallada'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    return parser.parse_args()


# ============================================================================
# VALIDACIÓN DE ARGUMENTOS
# ============================================================================

def validar_argumentos(args) -> bool:
    """
    Valida que los argumentos sean correctos (solo modo CLI).

    Args:
        args: Namespace con argumentos parseados

    Returns:
        True si todo es válido

    Raises:
        SystemExit: Si hay errores de validación
    """
    errores = []

    # Validar que existan los archivos (solo en modo CLI)
    if args.asistencia and not Path(args.asistencia).exists():
        errores.append(f"Archivo de asistencia no encontrado: {args.asistencia}")

    if args.calificaciones and not Path(args.calificaciones).exists():
        errores.append(f"Archivo de calificaciones no encontrado: {args.calificaciones}")

    # Validar umbrales si se proporcionan
    if args.umbrales or args.umbrales_file:
        try:
            umbrales = cargar_umbrales_personalizados(
                json_string=args.umbrales,
                json_file=args.umbrales_file
            )
            validar_umbrales(umbrales)
        except Exception as e:
            errores.append(f"Error en umbrales personalizados: {e}")

    # Validar directorio de salida
    try:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errores.append(f"No se puede crear directorio de salida {args.output}: {e}")

    # Si hay errores, mostrar y salir
    if errores:
        logger.error("\n❌ ERRORES DE VALIDACIÓN:\n")
        for error in errores:
            logger.error(f"  • {error}")
        logger.error("\nUsa --help para ver la ayuda.\n")
        sys.exit(1)

    return True


# ============================================================================
# MODO GUI
# ============================================================================

def main_gui():
    """
    Inicia la aplicación en modo GUI.
    """
    from gui import run_app
    run_app()


# ============================================================================
# MODO CLI
# ============================================================================

def main_cli(args):
    """
    Función principal del programa.

    Orquesta todo el flujo:
    1. Carga de datos
    2. Análisis descriptivo y ML
    3. Generación de reportes
    """
    # Banner
    print_banner()

    # Parsear argumentos
    args = parse_arguments()

    # Configurar logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Modo verbose activado")

    # Validar argumentos
    validar_argumentos(args)

    try:
        # ====================================================================
        # 1. CARGAR UMBRALES
        # ====================================================================
        logger.info("⚙️  Cargando configuración...")

        if args.umbrales or args.umbrales_file:
            umbrales = cargar_umbrales_personalizados(
                json_string=args.umbrales,
                json_file=args.umbrales_file
            )
            logger.info("✓ Umbrales personalizados cargados")
        else:
            umbrales = UMBRALES
            logger.info("✓ Usando umbrales por defecto")

        # ====================================================================
        # 2. CARGAR DATOS
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("PASO 1: CARGA DE DATOS")
        logger.info("=" * 70 + "\n")

        df_asistencia, df_calificaciones, estadisticas = cargar_datos(
            args.asistencia,
            args.calificaciones,
            umbrales
        )

        # Filtrar por curso si se especifica
        if args.curso:
            logger.info(f"\n🔍 Filtrando por curso: {args.curso}")

            registros_asist_antes = len(df_asistencia)
            registros_calif_antes = len(df_calificaciones)

            df_asistencia = df_asistencia[df_asistencia['CursoID'] == args.curso]
            df_calificaciones = df_calificaciones[df_calificaciones['CursoID'] == args.curso]

            if len(df_asistencia) == 0 or len(df_calificaciones) == 0:
                logger.error(
                    f"\n❌ ERROR: No se encontraron datos para el curso '{args.curso}'\n"
                    f"Cursos disponibles en asistencia: {df_asistencia['CursoID'].unique()}\n"
                    f"Cursos disponibles en calificaciones: {df_calificaciones['CursoID'].unique()}\n"
                )
                sys.exit(1)

            logger.info(
                f"✓ Filtrado completado:\n"
                f"  - Asistencia: {registros_asist_antes} → {len(df_asistencia)} registros\n"
                f"  - Calificaciones: {registros_calif_antes} → {len(df_calificaciones)} registros"
            )

        # ====================================================================
        # 3. ANÁLISIS
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("PASO 2: ANÁLISIS")
        logger.info("=" * 70 + "\n")

        resultados = realizar_analisis_completo(
            df_asistencia,
            df_calificaciones,
            umbrales
        )

        # Desactivar ML si se solicita
        if args.no_ml:
            logger.info("⊘ Machine Learning desactivado por opción --no-ml")
            resultados['ml'] = None

        # ====================================================================
        # 4. GENERAR REPORTES
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("PASO 3: GENERACIÓN DE REPORTES")
        logger.info("=" * 70 + "\n")

        archivos = generar_reportes(
            resultados,
            output_dir=args.output,
            formato=args.formato
        )

        # ====================================================================
        # 5. RESUMEN FINAL
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70 + "\n")

        logger.info("📊 RESUMEN DEL ANÁLISIS:")
        logger.info(f"  • Total estudiantes: {resultados['resumen']['total_estudiantes']}")
        logger.info(f"  • Riesgo ALTO: {resultados['resumen']['estudiantes_riesgo_alto']}")
        logger.info(f"  • Riesgo MEDIO: {resultados['resumen']['estudiantes_riesgo_medio']}")
        logger.info(f"  • ALERTA: {resultados['resumen']['estudiantes_alerta']}")
        logger.info(f"  • ÓPTIMO: {resultados['resumen']['estudiantes_optimo']}")

        logger.info("\n📁 ARCHIVOS GENERADOS:")
        for tipo, ruta in archivos.items():
            logger.info(f"  • {tipo.upper()}: {ruta}")

        logger.info("\n" + "=" * 70)
        logger.info("¡Gracias por usar el Sistema de Análisis de Riesgo Académico!")
        logger.info("=" * 70 + "\n")

        return 0

    except KeyboardInterrupt:
        logger.error("\n\n⚠️  Proceso interrumpido por el usuario.\n")
        return 130

    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error("❌ ERROR INESPERADO")
        logger.error("=" * 70)
        logger.error(f"\n{str(e)}\n")

        if args.verbose:
            import traceback
            logger.error("\n📋 TRACEBACK COMPLETO:\n")
            traceback.print_exc()

        logger.error("\n" + "=" * 70)
        logger.error("Si el problema persiste, verifica:")
        logger.error("  1. Formato de los archivos CSV")
        logger.error("  2. Columnas requeridas presentes")
        logger.error("  3. Encodings correctos (UTF-8 recomendado)")
        logger.error("  4. Permisos de escritura en directorio de salida")
        logger.error("\nUsa --verbose para más detalles.")
        logger.error("=" * 70 + "\n")

        return 1


# ============================================================================
# FUNCIÓN PRINCIPAL (SELECTOR GUI/CLI)
# ============================================================================

def main():
    """
    Función principal que decide entre modo GUI o CLI.

    - Si no hay argumentos o se usa --gui: inicia modo GUI
    - Si hay argumentos de datos (--asistencia, --calificaciones): usa modo CLI
    """
    # Si no hay argumentos de línea de comando, iniciar GUI
    if len(sys.argv) == 1:
        logger.info("🖥️  Iniciando modo GUI...")
        main_gui()
        return 0

    # Parsear argumentos
    args = parse_arguments()

    # Determinar modo
    usar_gui = (
        args.gui or
        (not args.cli and not args.asistencia and not args.calificaciones)
    )

    if usar_gui:
        logger.info("🖥️  Iniciando modo GUI...")
        main_gui()
        return 0
    else:
        # Validar que se proporcionen los archivos en modo CLI
        if not args.asistencia or not args.calificaciones:
            logger.error(
                "❌ Error: Modo CLI requiere --asistencia y --calificaciones\n"
                "Usa --gui para modo gráfico o proporciona ambos archivos.\n"
            )
            sys.exit(1)

        logger.info("💻 Iniciando modo CLI...")
        return main_cli(args)


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == '__main__':
    sys.exit(main())
