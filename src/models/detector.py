"""Clase principal del detector de vehículos basado en YOLO.

Encapsula la carga del modelo, detección, visualización y exportación
de resultados usando Ultralytics YOLO.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class VisionVialDetector:
    """Detector de vehículos usando YOLO (Ultralytics).

    Attributes:
        config: Diccionario de configuración.
        model: Modelo YOLO cargado.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Inicializa el detector con la configuración proporcionada.

        Args:
            config: Configuración del proyecto (de load_config()).
        """
        self.config = config
        model_config = config.get("model", {})

        # Determinar ruta del modelo
        architecture = model_config.get("architecture", "yolo11n")
        pretrained = model_config.get("pretrained", True)

        if pretrained:
            # Usar modelo pre-entrenado de Ultralytics
            self.model = YOLO(f"{architecture}.pt")
        else:
            # Cargar modelo desde archivo local
            model_path = Path(config.get("paths", {}).get("models", "models"))
            weights = list(model_path.glob("*.pt"))
            if weights:
                # Usar el modelo más reciente
                latest = max(weights, key=lambda p: p.stat().st_mtime)
                self.model = YOLO(str(latest))
            else:
                print("⚠️  No se encontraron modelos entrenados. Usando pretrained.")
                self.model = YOLO(f"{architecture}.pt")

        self.conf_threshold = model_config.get("conf_threshold", 0.25)
        self.iou_threshold = model_config.get("iou_threshold", 0.45)
        self.img_size = model_config.get("img_size", 640)

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        """Detecta vehículos en una imagen.

        Args:
            image_path: Ruta a la imagen.

        Returns:
            Lista de detecciones, cada una con: clase, confianza, bbox.
        """
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                det = {
                    "clase": result.names[int(boxes.cls[i])],
                    "confianza": float(boxes.conf[i]),
                    "bbox": {
                        "x1": float(boxes.xyxy[i][0]),
                        "y1": float(boxes.xyxy[i][1]),
                        "x2": float(boxes.xyxy[i][2]),
                        "y2": float(boxes.xyxy[i][3]),
                    },
                }
                detections.append(det)

        return detections

    def detect_batch(self, image_dir: str) -> dict[str, list[dict[str, Any]]]:
        """Procesa todas las imágenes de un directorio.

        Args:
            image_dir: Ruta al directorio con imágenes.

        Returns:
            Diccionario {nombre_archivo: lista_detecciones}.
        """
        image_dir = Path(image_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        results = {}

        for img_file in sorted(image_dir.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            detections = self.detect(str(img_file))
            results[img_file.name] = detections
            n = len(detections)
            print(f"  {img_file.name}: {n} detección{'es' if n != 1 else ''}")

        print(f"\n✅ Procesadas {len(results)} imágenes")
        return results

    def export_results(
        self,
        detections: dict[str, list[dict]],
        output_path: str,
        format: str = "json",
    ) -> str:
        """Guarda los resultados de detección en archivo.

        Args:
            detections: Resultados de detect_batch().
            output_path: Ruta de salida (sin extensión).
            format: Formato de salida ('json' o 'csv').

        Returns:
            Ruta al archivo generado.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            filepath = output.with_suffix(".json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(detections, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            filepath = output.with_suffix(".csv")
            rows = []
            for img_name, dets in detections.items():
                for det in dets:
                    rows.append({
                        "imagen": img_name,
                        "clase": det["clase"],
                        "confianza": det["confianza"],
                        "x1": det["bbox"]["x1"],
                        "y1": det["bbox"]["y1"],
                        "x2": det["bbox"]["x2"],
                        "y2": det["bbox"]["y2"],
                    })
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Formato no soportado: {format}")

        print(f"✅ Resultados guardados en: {filepath}")
        return str(filepath)

    def visualize(
        self,
        image_path: str,
        detections: list[dict] | None = None,
        output_path: str | None = None,
    ) -> np.ndarray:
        """Genera imagen con bounding boxes dibujados.

        Args:
            image_path: Ruta a la imagen original.
            detections: Lista de detecciones (si None, ejecuta detect()).
            output_path: Ruta opcional para guardar la imagen anotada.

        Returns:
            Imagen con bounding boxes como array numpy.
        """
        if detections is None:
            detections = self.detect(image_path)

        image = cv2.imread(image_path)

        # Colores por clase
        colors = {
            "car": (0, 255, 0),
            "truck": (255, 0, 0),
            "bus": (0, 0, 255),
            "motorcycle": (255, 255, 0),
            "ambulance": (0, 255, 255),
        }
        default_color = (128, 128, 128)

        for det in detections:
            bbox = det["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            clase = det["clase"]
            conf = det["confianza"]
            color = colors.get(clase, default_color)

            # Dibujar bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Dibujar etiqueta
            label = f"{clase} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1
            )
            cv2.putText(
                image, label, (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, image)
            print(f"✅ Imagen anotada guardada en: {output_path}")

        return image

    def get_stats(self, detections: dict[str, list[dict]] | list[dict]) -> dict:
        """Genera resumen estadístico de las detecciones.

        Args:
            detections: Resultados de detect() o detect_batch().

        Returns:
            Diccionario con conteo y confianza promedio por clase.
        """
        # Normalizar entrada
        all_dets: list[dict] = []
        if isinstance(detections, dict):
            for dets in detections.values():
                all_dets.extend(dets)
        else:
            all_dets = detections

        stats: dict[str, dict] = {}
        for det in all_dets:
            clase = det["clase"]
            if clase not in stats:
                stats[clase] = {"conteo": 0, "confianzas": []}
            stats[clase]["conteo"] += 1
            stats[clase]["confianzas"].append(det["confianza"])

        # Calcular promedios
        resumen = {}
        for clase, data in stats.items():
            resumen[clase] = {
                "conteo": data["conteo"],
                "confianza_promedio": sum(data["confianzas"]) / len(data["confianzas"]),
                "confianza_min": min(data["confianzas"]),
                "confianza_max": max(data["confianzas"]),
            }

        resumen["_total"] = {
            "detecciones": len(all_dets),
            "clases_detectadas": len(stats),
            "timestamp": datetime.now().isoformat(),
        }

        return resumen

    def train(
        self,
        dataset_yaml: str | None = None,
    ) -> Any:
        """Entrena el modelo con la configuración actual.

        Args:
            dataset_yaml: Ruta al YAML del dataset. Si None, usa config de training.

        Returns:
            Resultados del entrenamiento de Ultralytics.
        """
        training_config = self.config.get("training", {})

        train_args = {
            "data": dataset_yaml or training_config.get("dataset", ""),
            "epochs": training_config.get("epochs", 50),
            "batch": training_config.get("batch_size", 16),
            "imgsz": training_config.get("img_size", self.img_size),
            "patience": training_config.get("patience", 10),
            "save_period": training_config.get("save_period", 10),
            "project": training_config.get("project", "results"),
            "name": training_config.get("name_run", "train"),
        }

        print(f"🚀 Iniciando entrenamiento: {training_config.get('name', 'sin nombre')}")
        results = self.model.train(**train_args)
        print("✅ Entrenamiento completado")

        return results
