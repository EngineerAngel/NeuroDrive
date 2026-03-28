"""Funciones para exportar resultados de detección a diferentes formatos."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def detections_to_csv(
    detections: dict[str, list[dict]],
    output_path: str,
    include_metadata: bool = True,
) -> str:
    """Exporta detecciones a archivo CSV.

    Args:
        detections: Diccionario {imagen: lista_detecciones}.
        output_path: Ruta del archivo CSV de salida.
        include_metadata: Incluir columna de timestamp.

    Returns:
        Ruta al archivo generado.
    """
    rows = []
    for img_name, dets in detections.items():
        for det in dets:
            row = {
                "imagen": img_name,
                "clase": det["clase"],
                "confianza": det["confianza"],
                "x1": det["bbox"]["x1"],
                "y1": det["bbox"]["y1"],
                "x2": det["bbox"]["x2"],
                "y2": det["bbox"]["y2"],
            }
            if include_metadata:
                row["timestamp"] = datetime.now().isoformat()
            rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ CSV exportado: {output_path} ({len(rows)} detecciones)")
    return output_path


def detections_to_json(
    detections: dict[str, list[dict]],
    output_path: str,
    pretty: bool = True,
) -> str:
    """Exporta detecciones a archivo JSON.

    Args:
        detections: Diccionario {imagen: lista_detecciones}.
        output_path: Ruta del archivo JSON de salida.
        pretty: Formatear con indentación.

    Returns:
        Ruta al archivo generado.
    """
    output = {
        "proyecto": "NeuroDrive",
        "timestamp": datetime.now().isoformat(),
        "total_imagenes": len(detections),
        "total_detecciones": sum(len(d) for d in detections.values()),
        "resultados": detections,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2 if pretty else None, ensure_ascii=False)

    print(f"✅ JSON exportado: {output_path}")
    return output_path


def generate_report(
    detections: dict[str, list[dict]],
    stats: dict,
    output_path: str,
) -> str:
    """Genera un reporte resumido en texto plano.

    Args:
        detections: Resultados de detección.
        stats: Estadísticas de get_stats().
        output_path: Ruta del archivo de salida.

    Returns:
        Ruta al archivo generado.
    """
    lines = [
        "=" * 60,
        "REPORTE DE DETECCIÓN — NeuroDrive",
        "=" * 60,
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Imágenes procesadas: {len(detections)}",
        "",
    ]

    # Resumen por clase
    total_info = stats.get("_total", {})
    lines.append(f"Total de detecciones: {total_info.get('detecciones', 0)}")
    lines.append(f"Clases detectadas: {total_info.get('clases_detectadas', 0)}")
    lines.append("")
    lines.append("Detalle por clase:")
    lines.append("-" * 40)

    for clase, data in stats.items():
        if clase.startswith("_"):
            continue
        lines.append(
            f"  {clase:15s} | "
            f"Conteo: {data['conteo']:4d} | "
            f"Conf. promedio: {data['confianza_promedio']:.3f}"
        )

    lines.append("")
    lines.append("Detalle por imagen:")
    lines.append("-" * 40)

    for img_name, dets in detections.items():
        lines.append(f"  {img_name}: {len(dets)} detecciones")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Reporte generado: {output_path}")
    return output_path
