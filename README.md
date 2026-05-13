# Trial Matcher

Trial Matcher es un agente de Inteligencia Artificial que conecta automáticamente perfiles de pacientes con ensayos clínicos mediante la evaluación semántica de criterios médicos. El sistema recupera ensayos de ClinicalTrials.gov y genera un ranking justificado de elegibilidad de forma no interactiva.

## 1. Dependencias y Configuración

El proyecto requiere **Python >=3.11, <3.14**.

### Entorno virtual e instalación
```powershell
git clone https://github.com/aminelalam/Clinical_trial.git
cd Clinical_trial
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### Configuración de Credenciales
Copia la plantilla y edita el archivo `.env` con tus claves reales de LLM (ej. OpenAI/Azure):
```powershell
copy .env.example .env
```

### Descarga de Datos e Índices (TREC 2021)
Para reproducir los resultados, descarga el *snapshot* clínico y construye el índice BM25:
```powershell
bash scripts/download_trec.sh
python scripts\download_trec_ct_snapshot.py
python scripts\build_fielded_bm25_index.py --ctgov-dir data\trec_ct\clinical_trials_2021_04_27 --output-dir data\indices\bm25_trec2021_fielded --input-format ctgov-legacy-xml --write-index-manifest
```

## 2. Comandos de Ejecución

### Ejecución de la entrega final (Reproducción completa)
Para procesar los 75 pacientes del benchmark, evaluarlos y generar tanto las predicciones como los *dossiers*:
```powershell
python scripts\run_final_delivery.py --overwrite
```
*Resultado principal: `results/experiments/FINAL_DELIVERY_TREC2021_P12_FULL_75/predictions.json` (Copias maestras en `entrega_final/`).*

### Comprobación rápida (Topic 75)
Para comprobar que la arquitectura funciona rápidamente sobre 1 solo paciente, sin generar los costosos dossiers/preguntas (`--metric-only`):
```powershell
python scripts\run_final_delivery.py --topic-ids 75 --metric-only --run-name CHECK_TOPIC_75 --overwrite
```

### Comandos de Evaluación Manual y Testing
Si deseas computar las métricas sobre las predicciones de forma manual o validar el código fuente:
```powershell
python eval\trec_eval.py --predictions entrega_final\predicciones_trec2021_final.json --qrels data\trec_ct\raw\qrels_2021.txt --output results\trec_metrics_check.json
python eval\eligibility_eval.py --predictions entrega_final\predicciones_trec2021_final.json --qrels data\trec_ct\raw\qrels_2021.txt --output results\eligibility_metrics_check.json
python -m pytest tests\unit -q
```
