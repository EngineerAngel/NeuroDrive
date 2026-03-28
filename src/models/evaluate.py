"""Funciones de evaluación y comparación de modelos.

Genera métricas, matrices de confusión y comparativas entre
diferentes versiones de modelos entrenados.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


def evaluate_model(
    model_path: str,
    dataset_yaml: str,
    img_size: int = 640,
    conf: float = 0.25,
) -> dict[str, Any]:
    """Evalúa un modelo contra un dataset y retorna métricas.

    Args:
        model_path: Ruta al archivo .pt del modelo.
        dataset_yaml: Ruta al YAML del dataset.
        img_size: Tamaño de imagen.
        conf: Umbral de confianza.

    Returns:
        Diccionario con métricas (mAP50, mAP50-95, precision, recall).
    """
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml, imgsz=img_size, conf=conf)

    metrics = {
        "modelo": Path(model_path).name,
        "dataset": Path(dataset_yaml).stem,
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    print(f"📊 Resultados de {metrics['modelo']}:")
    print(f"   mAP@50:    {metrics['mAP50']:.4f}")
    print(f"   mAP@50-95: {metrics['mAP50_95']:.4f}")
    print(f"   Precision:  {metrics['precision']:.4f}")
    print(f"   Recall:     {metrics['recall']:.4f}")

    return metrics


def compare_models(
    model_paths: list[str],
    dataset_yaml: str,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Compara múltiples modelos en el mismo dataset.

    Args:
        model_paths: Lista de rutas a modelos .pt.
        dataset_yaml: Ruta al YAML del dataset de evaluación.
        output_dir: Directorio para guardar resultados.

    Returns:
        DataFrame con métricas comparativas.
    """
    all_metrics = []

    for model_path in model_paths:
        print(f"\n{'='*50}")
        print(f"Evaluando: {Path(model_path).name}")
        print("=" * 50)
        metrics = evaluate_model(model_path, dataset_yaml)
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    # Guardar CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "comparativa_modelos.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Comparativa guardada en: {csv_path}")

    return df


def plot_metrics_comparison(
    df: pd.DataFrame,
    output_path: str = "results/comparativa_metricas.png",
) -> None:
    """Genera gráfica comparativa de métricas entre modelos.

    Args:
        df: DataFrame con métricas (de compare_models()).
        output_path: Ruta para guardar la gráfica.
    """
    metrics_cols = ["mAP50", "mAP50_95", "precision", "recall"]
    x = np.arange(len(df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics_cols):
        ax.bar(x + i * width, df[metric], width, label=metric)

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Valor")
    ax.set_title("Comparativa de Métricas entre Modelos")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["modelo"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Gráfica guardada en: {output_path}")
