# Trial Matcher

Trial Matcher es un agente de Inteligencia Artificial que conecta automáticamente perfiles de pacientes con ensayos clínicos mediante el análisis semántico de criterios médicos complejos. El sistema lee el perfil, recupera ensayos candidatos de ClinicalTrials.gov y genera un dossier justificando la elegibilidad de cada uno.

## 🚀 Cómo usar el proyecto

El proyecto está diseñado para funcionar en bloque de manera automatizada. Le das una lista de pacientes y te devuelve los ensayos clínicos recomendados para cada uno.

### Ejecutar el pipeline completo
Para ejecutar el análisis sobre el dataset de pruebas completo (75 pacientes) y generar las predicciones, ejecuta:
```powershell
python scripts\run_final_delivery.py --overwrite
```

### ¿Qué resultados obtendrás?
Una vez termine la ejecución, en la carpeta `results/experiments/FINAL_DELIVERY_TREC2021_P12_FULL_75/` encontrarás:
1. `predictions.json`: El archivo final con los ensayos recomendados y puntuados para cada paciente.
2. **Dossiers**: Resúmenes médicos detallados que explican *por qué* el ensayo encaja con el paciente.
3. **Preguntas generadas**: Si faltaba información clave del paciente (ej. una prueba genética), la IA generará la pregunta exacta que el médico debe responder.

### Comprobación rápida
Si solo quieres probar que todo funciona analizando a un único paciente (Topic 75):
```powershell
python scripts\run_final_delivery.py --topic-ids 75 --metric-only --run-name PRUEBA_RAPIDA --overwrite
```

---

## 💻 Instalación (Para Desarrolladores)

Si quieres instalar el proyecto en tu propia máquina, sigue estos pasos (Requiere Python `>=3.11`):

1. **Clonar e instalar dependencias:**
```powershell
git clone https://github.com/aminelalam/Clinical_trial.git
cd Clinical_trial
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

2. **Configurar Credenciales:**
Copia la plantilla de configuración y añade tus claves de OpenAI/Azure.
```powershell
copy .env.example .env
```

3. **Descargar los datos clínicos:**
```powershell
bash scripts/download_trec.sh
python scripts\download_trec_ct_snapshot.py
python scripts\build_fielded_bm25_index.py --ctgov-dir data\trec_ct\clinical_trials_2021_04_27 --output-dir data\indices\bm25_trec2021_fielded --input-format ctgov-legacy-xml --write-index-manifest
```

---

## 📊 Rendimiento del Sistema

El agente ha sido evaluado bajo el estándar TREC Clinical Trials 2021, obteniendo una puntuación final de **52.62 / 100**, destacando en las siguientes métricas:
- **NDCG@10 (Calidad del ranking):** 0.520
- **Micro-F1 (Precisión evaluando criterios):** 0.499
- **Calidad de Preguntas Clínicas:** 0.913

Toda la documentación técnica y las memorias del proyecto se encuentran en la carpeta `entrega_final/`.

---

## 🤝 Contribuciones y Licencia

Cualquier mejora es bienvenida (especialmente en el módulo de *Dense Retrieval*). Puedes hacer un Pull Request asegurándote de que los tests pasen usando `python -m pytest tests\unit -q`.

Distribuido bajo la Licencia MIT. Libre para uso académico e investigación.
