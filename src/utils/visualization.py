"""Funciones de visualización para detecciones y resultados.

Genera imágenes anotadas, grids de resultados y gráficas de distribución.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Paleta de colores por clase (BGR para OpenCV)
CLASS_COLORS = {
    "car": (0, 255, 0),        # Verde
    "truck": (255, 0, 0),      # Azul
    "bus": (0, 0, 255),        # Rojo
    "motorcycle": (255, 255, 0),  # Cian
    "ambulance": (0, 255, 255),   # Amarillo
}
DEFAULT_COLOR = (128, 128, 128)  # Gris


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    show_confidence: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Dibuja bounding boxes y etiquetas sobre una imagen.

    Args:
        image: Imagen como array numpy (BGR).
        detections: Lista de detecciones con clase, confianza, bbox.
        show_confidence: Mostrar porcentaje de confianza.
        thickness: Grosor de las líneas del bbox.

    Returns:
        Imagen con las detecciones dibujadas.
    """
    annotated = image.copy()

    for det in detections:
        bbox = det["bbox"]
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        clase = det["clase"]
        conf = det["confianza"]
        color = CLASS_COLORS.get(clase, DEFAULT_COLOR)

        # Bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Etiqueta
        if show_confidence:
            label = f"{clase} {conf:.0%}"
        else:
            label = clase

        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - h - baseline - 4), (x1 + w + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return annotated


def create_results_grid(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    cols: int = 3,
    figsize: tuple[int, int] = (15, 10),
    output_path: str | None = None,
) -> None:
    """Genera un grid de imágenes con resultados.

    Args:
        images: Lista de imágenes (BGR).
        titles: Títulos opcionales para cada imagen.
        cols: Número de columnas.
        figsize: Tamaño de la figura.
        output_path: Ruta opcional para guardar.
    """
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes):
        if i < len(images):
            # Convertir BGR a RGB para matplotlib
            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Grid guardado en: {output_path}")
    plt.close()


def plot_class_distribution(
    stats: dict,
    output_path: str | None = None,
) -> None:
    """Genera gráfica de barras con distribución de clases detectadas.

    Args:
        stats: Estadísticas de get_stats() (sin la clave _total).
        output_path: Ruta opcional para guardar la gráfica.
    """
    # Filtrar _total si existe
    class_stats = {k: v for k, v in stats.items() if not k.startswith("_")}

    if not class_stats:
        print("⚠️  No hay clases para graficar")
        return

    clases = list(class_stats.keys())
    conteos = [class_stats[c]["conteo"] for c in clases]
    confianzas = [class_stats[c]["confianza_promedio"] for c in clases]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfica de conteo
    bars = ax1.bar(clases, conteos, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"])
    ax1.set_title("Conteo de Detecciones por Clase")
    ax1.set_ylabel("Cantidad")
    ax1.set_xlabel("Clase")
    for bar, count in zip(bars, conteos):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", fontsize=10)

    # Gráfica de confianza promedio
    bars2 = ax2.bar(clases, confianzas, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"])
    ax2.set_title("Confianza Promedio por Clase")
    ax2.set_ylabel("Confianza")
    ax2.set_xlabel("Clase")
    ax2.set_ylim(0, 1)
    for bar, conf in zip(bars2, confianzas):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{conf:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Gráfica guardada en: {output_path}")
    plt.close()
