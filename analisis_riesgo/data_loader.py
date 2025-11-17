"""
Módulo para carga y validación de datos desde archivos CSV.

Este módulo se encarga de:
- Leer archivos CSV de asistencia y calificaciones
- Validar estructura y contenido de los datos
- Limpiar y normalizar datos
- Anonimizar información sensible (cumplimiento RGPD)
- Preparar DataFrames para análisis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List
import logging

from config import DATOS_CONFIG, MENSAJES

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES DE CARGA DE CSV
# ============================================================================

def detectar_delimitador_y_encoding(file_path: str) -> Tuple[str, str]:
    """
    Detecta automáticamente el delimitador y encoding de un CSV.

    Args:
        file_path: Ruta al archivo CSV

    Returns:
        Tupla (delimitador, encoding)

    Raises:
        FileNotFoundError: Si el archivo no existe
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    # Probar combinaciones de encoding y delimitador
    for encoding in DATOS_CONFIG['encodings']:
        for delimiter in DATOS_CONFIG['delimitadores_csv']:
            try:
                # Intentar leer las primeras líneas
                df = pd.read_csv(
                    file_path,
                    sep=delimiter,
                    encoding=encoding,
                    nrows=5
                )
                # Si tiene más de 1 columna, probablemente es correcto
                if len(df.columns) > 1:
                    logger.info(
                        f"Detectado: delimiter='{delimiter}', "
                        f"encoding='{encoding}'"
                    )
                    return delimiter, encoding
            except Exception:
                continue

    # Si no se detectó, usar valores por defecto
    logger.warning(
        "No se pudo detectar delimitador y encoding. "
        "Usando valores por defecto: ',', 'utf-8'"
    )
    return ',', 'utf-8'


def leer_csv_robusto(file_path: str,
                     delimiter: str = None,
                     encoding: str = None) -> pd.DataFrame:
    """
    Lee un CSV de forma robusta, manejando diferentes formatos.

    Args:
        file_path: Ruta al archivo CSV
        delimiter: Delimitador (auto-detecta si es None)
        encoding: Encoding (auto-detecta si es None)

    Returns:
        DataFrame con los datos cargados

    Raises:
        ValueError: Si no se puede leer el archivo
    """
    if delimiter is None or encoding is None:
        delimiter, encoding = detectar_delimitador_y_encoding(file_path)

    try:
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            encoding=encoding,
            skipinitialspace=True,  # Eliminar espacios iniciales
            na_values=['', 'NA', 'N/A', 'null', 'None'],  # Valores nulos
        )

        # Limpiar nombres de columnas (quitar espacios)
        df.columns = df.columns.str.strip()

        logger.info(f"CSV leído exitosamente: {file_path}")
        logger.info(f"  - Filas: {len(df)}, Columnas: {len(df.columns)}")

        return df

    except Exception as e:
        raise ValueError(
            f"Error al leer CSV {file_path}: {str(e)}\n"
            f"Delimiter: {delimiter}, Encoding: {encoding}"
        )


# ============================================================================
# VALIDACIÓN DE DATOS
# ============================================================================

def validar_columnas(df: pd.DataFrame,
                     columnas_requeridas: List[str],
                     nombre_dataset: str) -> None:
    """
    Valida que el DataFrame tenga todas las columnas requeridas.

    Args:
        df: DataFrame a validar
        columnas_requeridas: Lista de nombres de columnas requeridas
        nombre_dataset: Nombre del dataset (para mensajes de error)

    Raises:
        ValueError: Si faltan columnas requeridas
    """
    columnas_faltantes = set(columnas_requeridas) - set(df.columns)

    if columnas_faltantes:
        raise ValueError(
            f"Columnas faltantes en {nombre_dataset}: "
            f"{', '.join(columnas_faltantes)}\n"
            f"Columnas disponibles: {', '.join(df.columns)}"
        )

    logger.info(f"✓ Validación de columnas exitosa para {nombre_dataset}")


def anonimizar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas con información sensible para cumplir RGPD.

    Args:
        df: DataFrame con posibles datos sensibles

    Returns:
        DataFrame anonimizado
    """
    columnas_a_eliminar = []

    for col_sensible in DATOS_CONFIG['columnas_sensibles']:
        if col_sensible in df.columns:
            columnas_a_eliminar.append(col_sensible)

    if columnas_a_eliminar:
        df = df.drop(columns=columnas_a_eliminar)
        logger.warning(
            f"⚠️  Columnas sensibles eliminadas (RGPD): "
            f"{', '.join(columnas_a_eliminar)}"
        )

    return df


def convertir_fechas(df: pd.DataFrame, columna_fecha: str = 'Fecha') -> pd.DataFrame:
    """
    Convierte columna de fecha a datetime manejando múltiples formatos.

    Args:
        df: DataFrame con columna de fecha
        columna_fecha: Nombre de la columna de fecha

    Returns:
        DataFrame con fecha convertida
    """
    if columna_fecha not in df.columns:
        logger.warning(f"Columna '{columna_fecha}' no encontrada. Saltando conversión.")
        return df

    # Formatos comunes de fecha
    formatos_fecha = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%Y/%m/%d',
        '%d.%m.%Y',
    ]

    # Intentar conversión automática primero
    try:
        df[columna_fecha] = pd.to_datetime(
            df[columna_fecha],
            infer_datetime_format=True,
            errors='coerce'
        )
        logger.info(f"✓ Fechas convertidas exitosamente en '{columna_fecha}'")
        return df
    except Exception:
        pass

    # Si falla, probar formatos específicos
    for formato in formatos_fecha:
        try:
            df[columna_fecha] = pd.to_datetime(
                df[columna_fecha],
                format=formato,
                errors='coerce'
            )
            logger.info(
                f"✓ Fechas convertidas con formato '{formato}' "
                f"en '{columna_fecha}'"
            )
            return df
        except Exception:
            continue

    # Si todo falla, dejar como está
    logger.warning(
        f"⚠️  No se pudieron convertir todas las fechas en '{columna_fecha}'"
    )
    return df


# ============================================================================
# LIMPIEZA DE DATOS
# ============================================================================

def limpiar_asistencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza el DataFrame de asistencia.

    Args:
        df: DataFrame de asistencia

    Returns:
        DataFrame limpio
    """
    df = df.copy()

    # Convertir 'Presente' a booleano
    if 'Presente' in df.columns:
        # Manejar diferentes representaciones de verdadero/falso
        df['Presente'] = df['Presente'].fillna(False)
        if df['Presente'].dtype == 'object':
            df['Presente'] = df['Presente'].str.lower().isin(
                ['true', '1', 'sí', 'si', 'yes', 't', 'verdadero', 'x']
            )
        else:
            df['Presente'] = df['Presente'].astype(bool)

    # Convertir 'Retraso' a booleano si existe
    if 'Retraso' in df.columns:
        df['Retraso'] = df['Retraso'].fillna(False)
        if df['Retraso'].dtype == 'object':
            df['Retraso'] = df['Retraso'].str.lower().isin(
                ['true', '1', 'sí', 'si', 'yes', 't', 'verdadero', 'x']
            )
        else:
            df['Retraso'] = df['Retraso'].astype(bool)
    else:
        df['Retraso'] = False

    # Convertir 'FaltaJustificada' a booleano si existe
    if 'FaltaJustificada' in df.columns:
        df['FaltaJustificada'] = df['FaltaJustificada'].fillna(False)
        if df['FaltaJustificada'].dtype == 'object':
            df['FaltaJustificada'] = df['FaltaJustificada'].str.lower().isin(
                ['true', '1', 'sí', 'si', 'yes', 't', 'verdadero', 'x']
            )
        else:
            df['FaltaJustificada'] = df['FaltaJustificada'].astype(bool)
    else:
        df['FaltaJustificada'] = False

    # Convertir fechas
    df = convertir_fechas(df, 'Fecha')

    # Eliminar duplicados
    registros_antes = len(df)
    df = df.drop_duplicates(subset=['IDEstudiante', 'CursoID', 'Fecha'])
    registros_despues = len(df)

    if registros_antes != registros_despues:
        logger.warning(
            f"⚠️  {registros_antes - registros_despues} registros "
            f"duplicados eliminados de asistencia"
        )

    # Ordenar por fecha
    if 'Fecha' in df.columns and df['Fecha'].dtype == 'datetime64[ns]':
        df = df.sort_values('Fecha')

    logger.info(f"✓ Asistencia limpiada: {len(df)} registros")

    return df


def limpiar_calificaciones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza el DataFrame de calificaciones.

    Args:
        df: DataFrame de calificaciones

    Returns:
        DataFrame limpio
    """
    df = df.copy()

    # Convertir 'Nota' a numérico
    if 'Nota' in df.columns:
        df['Nota'] = pd.to_numeric(df['Nota'], errors='coerce')

        # Eliminar notas fuera de rango (0-10)
        registros_antes = len(df)
        df = df[(df['Nota'] >= 0) & (df['Nota'] <= 10)]
        registros_despues = len(df)

        if registros_antes != registros_despues:
            logger.warning(
                f"⚠️  {registros_antes - registros_despues} registros "
                f"con notas fuera de rango eliminados"
            )

        # Eliminar registros sin nota
        registros_antes = len(df)
        df = df.dropna(subset=['Nota'])
        registros_despues = len(df)

        if registros_antes != registros_despues:
            logger.warning(
                f"⚠️  {registros_antes - registros_despues} registros "
                f"sin nota eliminados"
            )

    # Normalizar nombres de competencias
    if 'Competencia' in df.columns:
        df['Competencia'] = df['Competencia'].str.upper().str.strip()

    # Convertir 'Peso' a numérico si existe
    if 'Peso' in df.columns:
        df['Peso'] = pd.to_numeric(df['Peso'], errors='coerce')
        df['Peso'] = df['Peso'].fillna(1.0)  # Peso por defecto = 1
    else:
        df['Peso'] = 1.0

    # Convertir fechas si existe
    if 'Fecha' in df.columns:
        df = convertir_fechas(df, 'Fecha')

    # Eliminar duplicados
    registros_antes = len(df)
    df = df.drop_duplicates(
        subset=['IDEstudiante', 'CursoID', 'Evaluacion', 'Competencia']
    )
    registros_despues = len(df)

    if registros_antes != registros_despues:
        logger.warning(
            f"⚠️  {registros_antes - registros_despues} registros "
            f"duplicados eliminados de calificaciones"
        )

    logger.info(f"✓ Calificaciones limpiadas: {len(df)} registros")

    return df


# ============================================================================
# FUNCIONES INDIVIDUALES DE CARGA (para uso desde GUI)
# ============================================================================

def cargar_asistencia(asistencia_path: str) -> pd.DataFrame:
    """
    Carga solo el archivo de asistencia.

    Args:
        asistencia_path: Ruta al CSV de asistencia

    Returns:
        DataFrame de asistencia limpio y validado

    Raises:
        ValueError: Si los datos no son válidos
        FileNotFoundError: Si el archivo no existe
    """
    logger.info("📊 Cargando datos de ASISTENCIA...")

    df_asistencia = leer_csv_robusto(asistencia_path)

    # Validar columnas requeridas
    validar_columnas(
        df_asistencia,
        DATOS_CONFIG['columnas_asistencia'],
        'Asistencia'
    )

    # Anonimizar
    df_asistencia = anonimizar_datos(df_asistencia)

    # Limpiar
    df_asistencia = limpiar_asistencia(df_asistencia)

    logger.info(f"✓ Asistencia cargada: {len(df_asistencia)} registros")

    return df_asistencia


def cargar_calificaciones(calificaciones_path: str) -> pd.DataFrame:
    """
    Carga solo el archivo de calificaciones.

    Args:
        calificaciones_path: Ruta al CSV de calificaciones

    Returns:
        DataFrame de calificaciones limpio y validado

    Raises:
        ValueError: Si los datos no son válidos
        FileNotFoundError: Si el archivo no existe
    """
    logger.info("📊 Cargando datos de CALIFICACIONES...")

    df_calificaciones = leer_csv_robusto(calificaciones_path)

    # Validar columnas requeridas
    validar_columnas(
        df_calificaciones,
        DATOS_CONFIG['columnas_calificaciones'],
        'Calificaciones'
    )

    # Anonimizar
    df_calificaciones = anonimizar_datos(df_calificaciones)

    # Limpiar
    df_calificaciones = limpiar_calificaciones(df_calificaciones)

    logger.info(f"✓ Calificaciones cargadas: {len(df_calificaciones)} registros")

    return df_calificaciones


def validar_datos(df_asistencia: pd.DataFrame,
                  df_calificaciones: pd.DataFrame) -> None:
    """
    Valida la consistencia entre los datasets de asistencia y calificaciones.

    Args:
        df_asistencia: DataFrame de asistencia
        df_calificaciones: DataFrame de calificaciones

    Raises:
        ValueError: Si no hay datos en común entre los datasets
    """
    logger.info("🔍 Validando consistencia entre datasets...")

    # Verificar que haya estudiantes en común
    estudiantes_asist = set(df_asistencia['IDEstudiante'].unique())
    estudiantes_calif = set(df_calificaciones['IDEstudiante'].unique())
    estudiantes_comunes = estudiantes_asist & estudiantes_calif

    if not estudiantes_comunes:
        raise ValueError(
            "NO hay estudiantes en común entre asistencia y calificaciones. "
            "Verificar IDs de estudiantes."
        )
    else:
        logger.info(
            f"✓ {len(estudiantes_comunes)} estudiantes en común encontrados"
        )

    # Verificar que haya cursos en común
    cursos_asist = set(df_asistencia['CursoID'].unique())
    cursos_calif = set(df_calificaciones['CursoID'].unique())
    cursos_comunes = cursos_asist & cursos_calif

    if not cursos_comunes:
        logger.warning(
            "⚠️  NO hay cursos en común entre asistencia y calificaciones. "
            "Verificar IDs de cursos."
        )
    else:
        logger.info(
            f"✓ {len(cursos_comunes)} cursos en común encontrados"
        )

    logger.info("✓ Validación completada")


# ============================================================================
# FUNCIÓN PRINCIPAL DE CARGA
# ============================================================================

def cargar_datos(asistencia_path: str,
                calificaciones_path: str,
                umbrales: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Carga y valida los datos de asistencia y calificaciones.

    Esta es la función principal que orquesta todo el proceso de carga,
    validación, limpieza y anonimización de datos.

    Args:
        asistencia_path: Ruta al CSV de asistencia
        calificaciones_path: Ruta al CSV de calificaciones
        umbrales: Diccionario de umbrales (opcional)

    Returns:
        Tupla (df_asistencia, df_calificaciones, estadisticas)

    Raises:
        ValueError: Si los datos no son válidos
        FileNotFoundError: Si los archivos no existen
    """
    # Mostrar advertencia RGPD
    logger.info("\n" + "=" * 70)
    logger.info(MENSAJES['advertencia_rgpd'])
    logger.info("=" * 70 + "\n")

    estadisticas = {
        'fecha_carga': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'archivos': {
            'asistencia': asistencia_path,
            'calificaciones': calificaciones_path,
        }
    }

    # ========================================================================
    # 1. CARGAR ASISTENCIA
    # ========================================================================
    logger.info("📊 Cargando datos de ASISTENCIA...")

    df_asistencia = leer_csv_robusto(asistencia_path)

    # Validar columnas requeridas
    validar_columnas(
        df_asistencia,
        DATOS_CONFIG['columnas_asistencia'],
        'Asistencia'
    )

    # Anonimizar
    df_asistencia = anonimizar_datos(df_asistencia)

    # Limpiar
    df_asistencia = limpiar_asistencia(df_asistencia)

    # Estadísticas de asistencia
    estadisticas['asistencia'] = {
        'registros': len(df_asistencia),
        'estudiantes_unicos': df_asistencia['IDEstudiante'].nunique(),
        'cursos_unicos': df_asistencia['CursoID'].nunique(),
    }

    if 'Fecha' in df_asistencia.columns and df_asistencia['Fecha'].dtype == 'datetime64[ns]':
        estadisticas['asistencia']['fecha_min'] = df_asistencia['Fecha'].min()
        estadisticas['asistencia']['fecha_max'] = df_asistencia['Fecha'].max()

    # ========================================================================
    # 2. CARGAR CALIFICACIONES
    # ========================================================================
    logger.info("\n📊 Cargando datos de CALIFICACIONES...")

    df_calificaciones = leer_csv_robusto(calificaciones_path)

    # Validar columnas requeridas
    validar_columnas(
        df_calificaciones,
        DATOS_CONFIG['columnas_calificaciones'],
        'Calificaciones'
    )

    # Anonimizar
    df_calificaciones = anonimizar_datos(df_calificaciones)

    # Limpiar
    df_calificaciones = limpiar_calificaciones(df_calificaciones)

    # Estadísticas de calificaciones
    estadisticas['calificaciones'] = {
        'registros': len(df_calificaciones),
        'estudiantes_unicos': df_calificaciones['IDEstudiante'].nunique(),
        'cursos_unicos': df_calificaciones['CursoID'].nunique(),
        'evaluaciones_unicas': df_calificaciones['Evaluacion'].nunique(),
    }

    if 'Nota' in df_calificaciones.columns:
        estadisticas['calificaciones']['nota_media'] = df_calificaciones['Nota'].mean()
        estadisticas['calificaciones']['nota_min'] = df_calificaciones['Nota'].min()
        estadisticas['calificaciones']['nota_max'] = df_calificaciones['Nota'].max()

    # ========================================================================
    # 3. VALIDACIÓN CRUZADA
    # ========================================================================
    logger.info("\n🔍 Validando consistencia entre datasets...")

    # Verificar que haya estudiantes en común
    estudiantes_asist = set(df_asistencia['IDEstudiante'].unique())
    estudiantes_calif = set(df_calificaciones['IDEstudiante'].unique())
    estudiantes_comunes = estudiantes_asist & estudiantes_calif

    if not estudiantes_comunes:
        logger.warning(
            "⚠️  NO hay estudiantes en común entre asistencia y calificaciones. "
            "Verificar IDs."
        )
    else:
        logger.info(
            f"✓ {len(estudiantes_comunes)} estudiantes en común encontrados"
        )

    estadisticas['estudiantes_comunes'] = len(estudiantes_comunes)

    # Verificar que haya cursos en común
    cursos_asist = set(df_asistencia['CursoID'].unique())
    cursos_calif = set(df_calificaciones['CursoID'].unique())
    cursos_comunes = cursos_asist & cursos_calif

    if not cursos_comunes:
        logger.warning(
            "⚠️  NO hay cursos en común entre asistencia y calificaciones. "
            "Verificar IDs de cursos."
        )
    else:
        logger.info(
            f"✓ {len(cursos_comunes)} cursos en común encontrados"
        )

    estadisticas['cursos_comunes'] = len(cursos_comunes)

    # ========================================================================
    # 4. RESUMEN FINAL
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("✅ CARGA DE DATOS COMPLETADA")
    logger.info("=" * 70)
    logger.info(f"Asistencia: {estadisticas['asistencia']['registros']} registros")
    logger.info(f"  - {estadisticas['asistencia']['estudiantes_unicos']} estudiantes")
    logger.info(f"  - {estadisticas['asistencia']['cursos_unicos']} cursos")
    logger.info(f"\nCalificaciones: {estadisticas['calificaciones']['registros']} registros")
    logger.info(f"  - {estadisticas['calificaciones']['estudiantes_unicos']} estudiantes")
    logger.info(f"  - {estadisticas['calificaciones']['cursos_unicos']} cursos")
    logger.info(f"  - {estadisticas['calificaciones']['evaluaciones_unicas']} evaluaciones")
    logger.info(f"\nEstudiantes en común: {estadisticas['estudiantes_comunes']}")
    logger.info(f"Cursos en común: {estadisticas['cursos_comunes']}")
    logger.info("=" * 70 + "\n")

    return df_asistencia, df_calificaciones, estadisticas


# ============================================================================
# FUNCIÓN DE TEST
# ============================================================================

def test_data_loader():
    """
    Función de prueba básica para el data loader.
    Crea datos de ejemplo y prueba la carga.
    """
    import tempfile
    import os

    logger.info("🧪 Ejecutando tests de data_loader...")

    # Crear datos de prueba
    with tempfile.TemporaryDirectory() as tmpdir:
        # CSV de asistencia
        asistencia_csv = os.path.join(tmpdir, 'asistencia.csv')
        with open(asistencia_csv, 'w', encoding='utf-8') as f:
            f.write("IDEstudiante,CursoID,Fecha,Presente,Retraso,FaltaJustificada\n")
            f.write("EST001,3ESO-A,2024-01-10,True,False,False\n")
            f.write("EST001,3ESO-A,2024-01-12,False,False,True\n")
            f.write("EST002,3ESO-A,2024-01-10,True,True,False\n")

        # CSV de calificaciones
        calificaciones_csv = os.path.join(tmpdir, 'calificaciones.csv')
        with open(calificaciones_csv, 'w', encoding='utf-8') as f:
            f.write("IDEstudiante,CursoID,Evaluacion,Nota,Competencia,Peso\n")
            f.write("EST001,3ESO-A,Eval1,7.5,CE1,1.0\n")
            f.write("EST001,3ESO-A,Eval1,8.0,CE2,1.0\n")
            f.write("EST002,3ESO-A,Eval1,6.5,CE1,1.0\n")

        # Probar carga
        try:
            df_asist, df_calif, stats = cargar_datos(
                asistencia_csv,
                calificaciones_csv
            )

            assert len(df_asist) == 3, "Asistencia debe tener 3 registros"
            assert len(df_calif) == 3, "Calificaciones debe tener 3 registros"
            assert stats['estudiantes_comunes'] == 2, "Debe haber 2 estudiantes en común"

            logger.info("✅ Tests de data_loader PASADOS")
            return True

        except Exception as e:
            logger.error(f"❌ Test falló: {e}")
            return False


if __name__ == '__main__':
    # Ejecutar tests si se ejecuta directamente
    test_data_loader()
