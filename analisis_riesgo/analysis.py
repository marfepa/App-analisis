"""
Módulo de análisis descriptivo y machine learning para riesgo académico.

Este módulo implementa:
- Análisis descriptivo de asistencia y rendimiento
- Análisis agregado por curso
- Análisis individual por estudiante
- Ingeniería de features para ML
- Clustering (KMeans) para identificar grupos de riesgo
- Predicción y clasificación de riesgo académico
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from datetime import datetime

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats

from config import (
    UMBRALES,
    COMPETENCIAS,
    obtener_nivel_riesgo_asistencia,
    obtener_nivel_riesgo_nota,
    obtener_nivel_riesgo_combinado,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ANÁLISIS DE ASISTENCIA
# ============================================================================

def analizar_asistencia_estudiante(df_asistencia: pd.DataFrame,
                                   estudiante_id: str,
                                   curso_id: str = None,
                                   umbrales: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analiza la asistencia de un estudiante específico.

    Args:
        df_asistencia: DataFrame de asistencia
        estudiante_id: ID del estudiante
        curso_id: ID del curso (opcional, None para todos los cursos)
        umbrales: Diccionario de umbrales

    Returns:
        Diccionario con métricas de asistencia
    """
    if umbrales is None:
        umbrales = UMBRALES

    # Filtrar por estudiante (y curso si se especifica)
    mask = df_asistencia['IDEstudiante'] == estudiante_id
    if curso_id:
        mask &= df_asistencia['CursoID'] == curso_id
    df = df_asistencia[mask].copy()

    if len(df) == 0:
        return {
            'total_sesiones': 0,
            'asistencias': 0,
            'faltas': 0,
            'faltas_justificadas': 0,
            'faltas_injustificadas': 0,
            'retrasos': 0,
            'porcentaje_asistencia': 0.0,
            'nivel_riesgo': 'SIN_DATOS',
        }

    # Calcular métricas
    total_sesiones = len(df)
    asistencias = df['Presente'].sum()
    faltas = (~df['Presente']).sum()

    # Retrasos
    if 'Retraso' in df.columns:
        retrasos = df['Retraso'].sum()
    else:
        retrasos = 0

    # Faltas justificadas/injustificadas
    if 'FaltaJustificada' in df.columns:
        faltas_justificadas = df[~df['Presente'] & df['FaltaJustificada']].shape[0]
        faltas_injustificadas = df[~df['Presente'] & ~df['FaltaJustificada']].shape[0]
    else:
        faltas_justificadas = 0
        faltas_injustificadas = faltas

    # Porcentaje de asistencia
    porcentaje = (asistencias / total_sesiones * 100) if total_sesiones > 0 else 0.0

    # Nivel de riesgo
    nivel_riesgo = obtener_nivel_riesgo_asistencia(porcentaje, umbrales)

    # Patrón por día de semana (si hay fechas)
    patron_dias = {}
    if 'Fecha' in df.columns and df['Fecha'].dtype == 'datetime64[ns]':
        df['DiaSemana'] = df['Fecha'].dt.day_name()
        patron_dias = df.groupby('DiaSemana')['Presente'].agg(['sum', 'count']).to_dict('index')

        # Calcular porcentaje por día
        for dia, data in patron_dias.items():
            patron_dias[dia]['porcentaje'] = (
                data['sum'] / data['count'] * 100 if data['count'] > 0 else 0.0
            )

    # Tendencia temporal (últimas N sesiones vs primeras N)
    tendencia = 'ESTABLE'
    if total_sesiones >= 10:
        n = min(5, total_sesiones // 2)
        primeras = df.head(n)['Presente'].mean() * 100
        ultimas = df.tail(n)['Presente'].mean() * 100
        cambio = ultimas - primeras

        if cambio >= umbrales['tendencia_positiva']:
            tendencia = 'MEJORANDO'
        elif cambio <= umbrales['tendencia_negativa']:
            tendencia = 'EMPEORANDO'

    return {
        'total_sesiones': int(total_sesiones),
        'asistencias': int(asistencias),
        'faltas': int(faltas),
        'faltas_justificadas': int(faltas_justificadas),
        'faltas_injustificadas': int(faltas_injustificadas),
        'retrasos': int(retrasos),
        'porcentaje_asistencia': round(porcentaje, 2),
        'nivel_riesgo': nivel_riesgo,
        'patron_dias': patron_dias,
        'tendencia': tendencia,
    }


def analizar_asistencia_curso(df_asistencia: pd.DataFrame,
                              curso_id: str,
                              umbrales: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analiza la asistencia agregada de un curso.

    Args:
        df_asistencia: DataFrame de asistencia
        curso_id: ID del curso
        umbrales: Diccionario de umbrales

    Returns:
        Diccionario con métricas agregadas del curso
    """
    if umbrales is None:
        umbrales = UMBRALES

    df = df_asistencia[df_asistencia['CursoID'] == curso_id].copy()

    if len(df) == 0:
        return {'estudiantes': 0, 'sesiones': 0}

    # Calcular por estudiante
    estudiantes = df['IDEstudiante'].unique()
    metricas_estudiantes = []

    for est_id in estudiantes:
        metricas = analizar_asistencia_estudiante(df_asistencia, est_id, curso_id, umbrales)
        metricas_estudiantes.append(metricas)

    # Agregados
    porcentajes = [m['porcentaje_asistencia'] for m in metricas_estudiantes]

    return {
        'curso_id': curso_id,
        'estudiantes': len(estudiantes),
        'sesiones_totales': df.groupby('IDEstudiante').size().mean(),
        'porcentaje_medio': np.mean(porcentajes),
        'porcentaje_mediana': np.median(porcentajes),
        'porcentaje_std': np.std(porcentajes),
        'porcentaje_min': np.min(porcentajes),
        'porcentaje_max': np.max(porcentajes),
        'estudiantes_riesgo_alto': sum(1 for m in metricas_estudiantes if m['nivel_riesgo'] == 'ALTO'),
        'estudiantes_riesgo_medio': sum(1 for m in metricas_estudiantes if m['nivel_riesgo'] == 'MEDIO'),
        'estudiantes_alerta': sum(1 for m in metricas_estudiantes if m['nivel_riesgo'] == 'ALERTA'),
    }


# ============================================================================
# ANÁLISIS DE RENDIMIENTO
# ============================================================================

def analizar_rendimiento_estudiante(df_calificaciones: pd.DataFrame,
                                    estudiante_id: str,
                                    curso_id: str = None,
                                    umbrales: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analiza el rendimiento académico de un estudiante.

    Args:
        df_calificaciones: DataFrame de calificaciones
        estudiante_id: ID del estudiante
        curso_id: ID del curso (opcional)
        umbrales: Diccionario de umbrales

    Returns:
        Diccionario con métricas de rendimiento
    """
    if umbrales is None:
        umbrales = UMBRALES

    # Filtrar por estudiante (y curso si se especifica)
    mask = df_calificaciones['IDEstudiante'] == estudiante_id
    if curso_id:
        mask &= df_calificaciones['CursoID'] == curso_id
    df = df_calificaciones[mask].copy()

    if len(df) == 0:
        return {
            'n_evaluaciones': 0,
            'nota_media': 0.0,
            'nivel_riesgo': 'SIN_DATOS',
        }

    # Calcular nota media ponderada
    if 'Peso' in df.columns:
        nota_media = np.average(df['Nota'], weights=df['Peso'])
    else:
        nota_media = df['Nota'].mean()

    # Estadísticas básicas
    nota_min = df['Nota'].min()
    nota_max = df['Nota'].max()
    nota_std = df['Nota'].std()

    # Nivel de riesgo
    nivel_riesgo = obtener_nivel_riesgo_nota(nota_media, umbrales)

    # Análisis por competencia
    competencias_stats = {}
    if 'Competencia' in df.columns:
        for comp in df['Competencia'].unique():
            if pd.notna(comp):
                notas_comp = df[df['Competencia'] == comp]['Nota']
                competencias_stats[comp] = {
                    'nota_media': notas_comp.mean(),
                    'n_evaluaciones': len(notas_comp),
                }

    # Tendencia temporal (si hay fechas o evaluaciones ordenadas)
    tendencia = 'ESTABLE'
    if len(df) >= 3:
        # Usar índice como proxy temporal si no hay fechas
        if 'Fecha' in df.columns and df['Fecha'].dtype == 'datetime64[ns]':
            df = df.sort_values('Fecha')

        # Calcular tendencia con regresión lineal simple
        x = np.arange(len(df))
        y = df['Nota'].values
        if len(x) > 1 and np.std(y) > 0:
            slope, _, _, _, _ = stats.linregress(x, y)

            # Calcular cambio porcentual
            cambio_porcentual = (slope * len(x) / nota_media * 100) if nota_media > 0 else 0

            if cambio_porcentual >= umbrales['tendencia_positiva']:
                tendencia = 'MEJORANDO'
            elif cambio_porcentual <= umbrales['tendencia_negativa']:
                tendencia = 'EMPEORANDO'

    return {
        'n_evaluaciones': int(len(df)),
        'nota_media': round(nota_media, 2),
        'nota_min': round(nota_min, 2),
        'nota_max': round(nota_max, 2),
        'nota_std': round(nota_std, 2),
        'nivel_riesgo': nivel_riesgo,
        'competencias': competencias_stats,
        'tendencia': tendencia,
        'aprobado': nota_media >= umbrales['nota_aprobado'],
    }


def analizar_rendimiento_curso(df_calificaciones: pd.DataFrame,
                               curso_id: str,
                               umbrales: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analiza el rendimiento agregado de un curso.

    Args:
        df_calificaciones: DataFrame de calificaciones
        curso_id: ID del curso
        umbrales: Diccionario de umbrales

    Returns:
        Diccionario con métricas agregadas del curso
    """
    if umbrales is None:
        umbrales = UMBRALES

    df = df_calificaciones[df_calificaciones['CursoID'] == curso_id].copy()

    if len(df) == 0:
        return {'estudiantes': 0, 'evaluaciones': 0}

    # Calcular por estudiante
    estudiantes = df['IDEstudiante'].unique()
    metricas_estudiantes = []

    for est_id in estudiantes:
        metricas = analizar_rendimiento_estudiante(df_calificaciones, est_id, curso_id, umbrales)
        metricas_estudiantes.append(metricas)

    # Agregados
    notas = [m['nota_media'] for m in metricas_estudiantes if m['n_evaluaciones'] > 0]

    resultado = {
        'curso_id': curso_id,
        'estudiantes': len(estudiantes),
        'evaluaciones_totales': len(df),
        'nota_media': np.mean(notas) if notas else 0.0,
        'nota_mediana': np.median(notas) if notas else 0.0,
        'nota_std': np.std(notas) if notas else 0.0,
        'nota_min': np.min(notas) if notas else 0.0,
        'nota_max': np.max(notas) if notas else 0.0,
        'tasa_aprobados': sum(1 for m in metricas_estudiantes if m.get('aprobado', False)) / len(estudiantes) * 100 if estudiantes.size > 0 else 0.0,
        'estudiantes_riesgo_alto': sum(1 for m in metricas_estudiantes if m['nivel_riesgo'] == 'ALTO'),
        'estudiantes_riesgo_medio': sum(1 for m in metricas_estudiantes if m['nivel_riesgo'] == 'MEDIO'),
        'estudiantes_alerta': sum(1 for m in metricas_estudiantes if m['nivel_riesgo'] == 'ALERTA'),
    }

    # Análisis por competencia
    if 'Competencia' in df.columns:
        competencias_stats = {}
        for comp in df['Competencia'].unique():
            if pd.notna(comp):
                notas_comp = df[df['Competencia'] == comp]['Nota']
                competencias_stats[comp] = {
                    'nota_media': notas_comp.mean(),
                    'nota_std': notas_comp.std(),
                    'n_evaluaciones': len(notas_comp),
                }
        resultado['competencias'] = competencias_stats

    return resultado


# ============================================================================
# INGENIERÍA DE FEATURES PARA MACHINE LEARNING
# ============================================================================

def crear_features_ml(df_asistencia: pd.DataFrame,
                     df_calificaciones: pd.DataFrame,
                     umbrales: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Crea features para machine learning a partir de asistencia y calificaciones.

    Args:
        df_asistencia: DataFrame de asistencia
        df_calificaciones: DataFrame de calificaciones
        umbrales: Diccionario de umbrales

    Returns:
        DataFrame con features por estudiante
    """
    if umbrales is None:
        umbrales = UMBRALES

    logger.info("🔧 Creando features para Machine Learning...")

    # Lista para almacenar features de cada estudiante
    features_list = []

    # Obtener estudiantes únicos
    estudiantes = set(df_asistencia['IDEstudiante'].unique()) | set(df_calificaciones['IDEstudiante'].unique())

    for est_id in estudiantes:
        features = {'IDEstudiante': est_id}

        # ====================================================================
        # FEATURES DE ASISTENCIA
        # ====================================================================
        asist_metrics = analizar_asistencia_estudiante(df_asistencia, est_id, umbrales=umbrales)

        features['asist_porcentaje'] = asist_metrics['porcentaje_asistencia']
        features['asist_total_sesiones'] = asist_metrics['total_sesiones']
        features['asist_faltas'] = asist_metrics['faltas']
        features['asist_faltas_injustificadas'] = asist_metrics['faltas_injustificadas']
        features['asist_retrasos'] = asist_metrics['retrasos']

        # Ratios
        if asist_metrics['total_sesiones'] > 0:
            features['asist_ratio_faltas'] = asist_metrics['faltas'] / asist_metrics['total_sesiones']
            features['asist_ratio_retrasos'] = asist_metrics['retrasos'] / asist_metrics['total_sesiones']
        else:
            features['asist_ratio_faltas'] = 0.0
            features['asist_ratio_retrasos'] = 0.0

        # Indicador de tendencia
        features['asist_tendencia_mejorando'] = 1 if asist_metrics['tendencia'] == 'MEJORANDO' else 0
        features['asist_tendencia_empeorando'] = 1 if asist_metrics['tendencia'] == 'EMPEORANDO' else 0

        # ====================================================================
        # FEATURES DE RENDIMIENTO
        # ====================================================================
        rend_metrics = analizar_rendimiento_estudiante(df_calificaciones, est_id, umbrales=umbrales)

        features['rend_nota_media'] = rend_metrics['nota_media']
        features['rend_nota_std'] = rend_metrics.get('nota_std', 0.0)
        features['rend_n_evaluaciones'] = rend_metrics['n_evaluaciones']

        # Indicadores
        features['rend_aprobado'] = 1 if rend_metrics.get('aprobado', False) else 0
        features['rend_tendencia_mejorando'] = 1 if rend_metrics['tendencia'] == 'MEJORANDO' else 0
        features['rend_tendencia_empeorando'] = 1 if rend_metrics['tendencia'] == 'EMPEORANDO' else 0

        # Features por competencia
        competencias_metrics = rend_metrics.get('competencias', {})
        for comp_id in ['CE1', 'CE2', 'CE3', 'CE4', 'CE5']:
            if comp_id in competencias_metrics:
                features[f'comp_{comp_id}_nota'] = competencias_metrics[comp_id]['nota_media']
            else:
                features[f'comp_{comp_id}_nota'] = 0.0

        # ====================================================================
        # FEATURES COMBINADAS
        # ====================================================================

        # Ratio asistencia/rendimiento
        if features['asist_porcentaje'] > 0:
            features['ratio_rend_asist'] = features['rend_nota_media'] / (features['asist_porcentaje'] / 10)
        else:
            features['ratio_rend_asist'] = 0.0

        # Score de riesgo manual (basado en umbrales)
        score_riesgo = 0
        if features['asist_porcentaje'] < umbrales['asistencia_riesgo']:
            score_riesgo += 3
        elif features['asist_porcentaje'] < umbrales['asistencia_alerta']:
            score_riesgo += 2

        if features['rend_nota_media'] < umbrales['nota_riesgo']:
            score_riesgo += 3
        elif features['rend_nota_media'] < umbrales['nota_alerta']:
            score_riesgo += 2

        features['score_riesgo_manual'] = score_riesgo

        features_list.append(features)

    df_features = pd.DataFrame(features_list)

    # Manejar valores faltantes
    df_features = df_features.fillna(0)

    logger.info(f"✓ Features creados para {len(df_features)} estudiantes")
    logger.info(f"  - {len(df_features.columns) - 1} features por estudiante")

    return df_features


# ============================================================================
# MACHINE LEARNING: CLUSTERING
# ============================================================================

def entrenar_modelo_clustering(df_features: pd.DataFrame,
                               umbrales: Dict[str, Any] = None) -> Tuple[KMeans, StandardScaler, np.ndarray, pd.DataFrame]:
    """
    Entrena modelo de clustering (KMeans) para identificar grupos de riesgo.

    Args:
        df_features: DataFrame con features por estudiante
        umbrales: Diccionario de umbrales

    Returns:
        Tupla (modelo, scaler, labels, df_features_con_clusters)
    """
    if umbrales is None:
        umbrales = UMBRALES

    logger.info("🤖 Entrenando modelo de Machine Learning (KMeans)...")

    # Verificar que hay suficientes muestras
    if len(df_features) < umbrales['ml_min_samples']:
        logger.warning(
            f"⚠️  Pocas muestras para ML ({len(df_features)} < {umbrales['ml_min_samples']}). "
            f"Resultados pueden no ser confiables."
        )

    # Seleccionar features para clustering (excluir ID)
    feature_cols = [col for col in df_features.columns if col != 'IDEstudiante']
    X = df_features[feature_cols].values

    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determinar número óptimo de clusters (probar 2-5)
    best_n_clusters = umbrales['ml_n_clusters']
    best_score = -1

    if len(df_features) >= 10:  # Solo si hay suficientes muestras
        logger.info("  🔍 Buscando número óptimo de clusters...")
        for n in range(2, min(6, len(df_features))):
            kmeans_temp = KMeans(
                n_clusters=n,
                random_state=umbrales['ml_random_state'],
                n_init=10
            )
            labels_temp = kmeans_temp.fit_predict(X_scaled)

            # Calcular silhouette score (solo si n < n_samples)
            if n < len(df_features):
                score = silhouette_score(X_scaled, labels_temp)
                logger.info(f"    - n_clusters={n}: silhouette={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_n_clusters = n

        logger.info(f"  ✓ Mejor n_clusters: {best_n_clusters} (silhouette={best_score:.3f})")

    # Entrenar modelo final
    modelo = KMeans(
        n_clusters=best_n_clusters,
        random_state=umbrales['ml_random_state'],
        n_init=10
    )
    labels = modelo.fit_predict(X_scaled)

    # Asignar labels a DataFrame
    df_resultado = df_features.copy()
    df_resultado['Cluster'] = labels

    # ========================================================================
    # INTERPRETAR CLUSTERS: Asignar nivel de riesgo a cada cluster
    # ========================================================================

    # Calcular métricas promedio por cluster
    cluster_stats = []
    for cluster_id in range(best_n_clusters):
        mask = labels == cluster_id
        cluster_data = df_features[mask]

        stats_cluster = {
            'cluster': cluster_id,
            'n_estudiantes': mask.sum(),
            'asist_media': cluster_data['asist_porcentaje'].mean(),
            'nota_media': cluster_data['rend_nota_media'].mean(),
            'faltas_media': cluster_data['asist_faltas'].mean(),
            'score_riesgo_media': cluster_data['score_riesgo_manual'].mean(),
        }
        cluster_stats.append(stats_cluster)

    df_cluster_stats = pd.DataFrame(cluster_stats)

    # Ordenar clusters por score de riesgo (más alto = más riesgo)
    df_cluster_stats = df_cluster_stats.sort_values('score_riesgo_media', ascending=False)

    # Asignar niveles de riesgo
    niveles = ['ALTO', 'MEDIO', 'ALERTA'] if best_n_clusters >= 3 else ['ALTO', 'MEDIO']
    if best_n_clusters > len(niveles):
        niveles.extend(['OPTIMO'] * (best_n_clusters - len(niveles)))

    cluster_to_nivel = {}
    for i, (_, row) in enumerate(df_cluster_stats.iterrows()):
        cluster_to_nivel[row['cluster']] = niveles[i] if i < len(niveles) else 'OPTIMO'

    # Agregar nivel de riesgo al DataFrame
    df_resultado['NivelRiesgoML'] = df_resultado['Cluster'].map(cluster_to_nivel)

    # Log de resultados
    logger.info(f"✓ Modelo entrenado exitosamente")
    logger.info(f"  - {best_n_clusters} clusters identificados")
    logger.info(f"\n  Distribución de clusters:")
    for _, row in df_cluster_stats.iterrows():
        nivel = cluster_to_nivel[row['cluster']]
        logger.info(
            f"    Cluster {int(row['cluster'])} ({nivel}): "
            f"{int(row['n_estudiantes'])} estudiantes, "
            f"Asist={row['asist_media']:.1f}%, "
            f"Nota={row['nota_media']:.1f}"
        )

    return modelo, scaler, labels, df_resultado


# ============================================================================
# ANÁLISIS COMPLETO
# ============================================================================

def realizar_analisis_completo(df_asistencia: pd.DataFrame,
                              df_calificaciones: pd.DataFrame,
                              umbrales: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Realiza el análisis completo: descriptivo + ML.

    Args:
        df_asistencia: DataFrame de asistencia
        df_calificaciones: DataFrame de calificaciones
        umbrales: Diccionario de umbrales

    Returns:
        Diccionario con todos los resultados del análisis
    """
    if umbrales is None:
        umbrales = UMBRALES

    logger.info("\n" + "=" * 70)
    logger.info("📊 INICIANDO ANÁLISIS COMPLETO")
    logger.info("=" * 70)

    resultados = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'umbrales': umbrales,
    }

    # ========================================================================
    # 1. ANÁLISIS POR CURSO
    # ========================================================================
    logger.info("\n🏫 Analizando cursos...")

    cursos = set(df_asistencia['CursoID'].unique()) | set(df_calificaciones['CursoID'].unique())
    analisis_cursos = []

    for curso_id in cursos:
        analisis_asist = analizar_asistencia_curso(df_asistencia, curso_id, umbrales)
        analisis_rend = analizar_rendimiento_curso(df_calificaciones, curso_id, umbrales)

        # Combinar
        analisis_curso = {
            'curso_id': curso_id,
            **analisis_asist,
            **{f'rend_{k}': v for k, v in analisis_rend.items() if k != 'curso_id'}
        }
        analisis_cursos.append(analisis_curso)

    resultados['analisis_cursos'] = pd.DataFrame(analisis_cursos)
    logger.info(f"✓ {len(analisis_cursos)} cursos analizados")

    # ========================================================================
    # 2. ANÁLISIS INDIVIDUAL
    # ========================================================================
    logger.info("\n👤 Analizando estudiantes individuales...")

    estudiantes = set(df_asistencia['IDEstudiante'].unique()) | set(df_calificaciones['IDEstudiante'].unique())
    analisis_individuales = []

    for est_id in estudiantes:
        analisis_asist = analizar_asistencia_estudiante(df_asistencia, est_id, umbrales=umbrales)
        analisis_rend = analizar_rendimiento_estudiante(df_calificaciones, est_id, umbrales=umbrales)

        # Nivel de riesgo combinado
        nivel_riesgo_final = obtener_nivel_riesgo_combinado(
            analisis_asist['nivel_riesgo'],
            analisis_rend['nivel_riesgo']
        )

        analisis_individual = {
            'IDEstudiante': est_id,
            'nivel_riesgo_asistencia': analisis_asist['nivel_riesgo'],
            'nivel_riesgo_rendimiento': analisis_rend['nivel_riesgo'],
            'nivel_riesgo_final': nivel_riesgo_final,
            **{f'asist_{k}': v for k, v in analisis_asist.items() if k != 'nivel_riesgo' and k != 'patron_dias'},
            **{f'rend_{k}': v for k, v in analisis_rend.items() if k != 'nivel_riesgo' and k != 'competencias'},
        }
        analisis_individuales.append(analisis_individual)

    resultados['analisis_individual'] = pd.DataFrame(analisis_individuales)
    logger.info(f"✓ {len(analisis_individuales)} estudiantes analizados")

    # ========================================================================
    # 3. MACHINE LEARNING
    # ========================================================================
    logger.info("\n🤖 Aplicando Machine Learning...")

    df_features = crear_features_ml(df_asistencia, df_calificaciones, umbrales)

    if len(df_features) >= umbrales['ml_min_samples']:
        modelo, scaler, labels, df_con_clusters = entrenar_modelo_clustering(df_features, umbrales)

        resultados['ml'] = {
            'modelo': modelo,
            'scaler': scaler,
            'features': df_features,
            'predicciones': df_con_clusters,
        }

        # Combinar con análisis individual
        df_combined = resultados['analisis_individual'].merge(
            df_con_clusters[['IDEstudiante', 'Cluster', 'NivelRiesgoML']],
            on='IDEstudiante',
            how='left'
        )
        resultados['analisis_individual'] = df_combined

    else:
        logger.warning(
            f"⚠️  Insuficientes estudiantes para ML "
            f"({len(df_features)} < {umbrales['ml_min_samples']}). "
            f"Saltando análisis ML."
        )
        resultados['ml'] = None

    # ========================================================================
    # 4. RESUMEN GLOBAL
    # ========================================================================
    logger.info("\n📈 Generando resumen global...")

    resumen = {
        'total_estudiantes': len(estudiantes),
        'total_cursos': len(cursos),
        'estudiantes_riesgo_alto': (resultados['analisis_individual']['nivel_riesgo_final'] == 'ALTO').sum(),
        'estudiantes_riesgo_medio': (resultados['analisis_individual']['nivel_riesgo_final'] == 'MEDIO').sum(),
        'estudiantes_alerta': (resultados['analisis_individual']['nivel_riesgo_final'] == 'ALERTA').sum(),
        'estudiantes_optimo': (resultados['analisis_individual']['nivel_riesgo_final'] == 'OPTIMO').sum(),
    }

    # Porcentajes
    if resumen['total_estudiantes'] > 0:
        resumen['porcentaje_riesgo_alto'] = resumen['estudiantes_riesgo_alto'] / resumen['total_estudiantes'] * 100
        resumen['porcentaje_riesgo_medio'] = resumen['estudiantes_riesgo_medio'] / resumen['total_estudiantes'] * 100

    resultados['resumen'] = resumen

    # ========================================================================
    # FINALIZADO
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("✅ ANÁLISIS COMPLETADO")
    logger.info("=" * 70)
    logger.info(f"Total estudiantes: {resumen['total_estudiantes']}")
    logger.info(f"  - Riesgo ALTO: {resumen['estudiantes_riesgo_alto']}")
    logger.info(f"  - Riesgo MEDIO: {resumen['estudiantes_riesgo_medio']}")
    logger.info(f"  - ALERTA: {resumen['estudiantes_alerta']}")
    logger.info(f"  - ÓPTIMO: {resumen['estudiantes_optimo']}")
    logger.info("=" * 70 + "\n")

    return resultados


# ============================================================================
# FUNCIÓN DE TEST
# ============================================================================

def test_analysis():
    """Test básico del módulo de análisis."""
    logger.info("🧪 Ejecutando tests de analysis...")

    # Crear datos de prueba
    df_asist = pd.DataFrame({
        'IDEstudiante': ['EST001', 'EST001', 'EST002', 'EST002'] * 5,
        'CursoID': ['3ESO-A'] * 20,
        'Fecha': pd.date_range('2024-01-01', periods=20),
        'Presente': [True, True, False, True] * 5,
        'Retraso': [False] * 20,
        'FaltaJustificada': [False] * 20,
    })

    df_calif = pd.DataFrame({
        'IDEstudiante': ['EST001', 'EST001', 'EST002', 'EST002'],
        'CursoID': ['3ESO-A'] * 4,
        'Evaluacion': ['Eval1', 'Eval2', 'Eval1', 'Eval2'],
        'Nota': [7.5, 8.0, 5.0, 4.5],
        'Competencia': ['CE1', 'CE2', 'CE1', 'CE2'],
        'Peso': [1.0] * 4,
    })

    try:
        # Test análisis completo
        resultados = realizar_analisis_completo(df_asist, df_calif)

        assert 'analisis_cursos' in resultados
        assert 'analisis_individual' in resultados
        assert len(resultados['analisis_individual']) == 2

        logger.info("✅ Tests de analysis PASADOS")
        return True

    except Exception as e:
        logger.error(f"❌ Test falló: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_analysis()
