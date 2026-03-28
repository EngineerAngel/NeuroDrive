.PHONY: install train detect api test clean help

help: ## Mostrar esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Crear entorno virtual e instalar dependencias
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✅ Entorno listo. Activa con: source venv/bin/activate"

train: ## Entrenar modelo con configuración baseline
	python -c "from src.models.detector import VisionVialDetector; from src.config import load_config; \
		config = load_config('configs/training/baseline.yaml'); \
		detector = VisionVialDetector(config); \
		detector.train()"

detect: ## Correr detección en data/samples/
	python scripts/quick_detect.py --image data/samples/ --output results/

api: ## Levantar API en localhost:8000
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Correr tests con pytest
	pytest tests/ -v

clean: ## Limpiar resultados y archivos temporales
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf runs/ wandb/
	@echo "🧹 Limpieza completada"
