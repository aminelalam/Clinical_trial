# Trial Matcher

Agente no interactivo para matching paciente-ensayo clinico, evaluado sobre
TREC Clinical Trials 2021. El sistema recupera ensayos de ClinicalTrials.gov,
evalua criterios de elegibilidad, ordena candidatos, genera preguntas de
informacion faltante y construye dossiers de preseleccion listos para revision
medica.

## Resultado Final

La entrega final esta en `entrega_final/` y corresponde a la corrida P12 sobre
los 75 topics de TREC Clinical Trials 2021.

| Componente | Metrica | Valor |
| --- | --- | ---: |
| T1 retrieval | Recall@20 | 0.094129 |
| T2 elegibilidad | Micro-F1 | 0.499281 |
| T3 ranking | NDCG@10 | 0.520406 |
| T4 preguntas | Calidad media | 0.913933 |
| T5 dossier | Completitud media | 0.903993 |
| Score hackaton | Formula ponderada | 0.526201 |

Score expresado como porcentaje: **52.62 / 100**.

Formula del reto:

```text
Score = 0.20 * Recall@20
      + 0.30 * Micro-F1
      + 0.25 * NDCG@10
      + 0.15 * calidad preguntas
      + 0.10 * completitud dossier
```

La mejora final P12 sube `NDCG@10` frente al baseline P11 de `0.517134` a
`0.520406` y `MRR` de `0.744704` a `0.763815`, sin cambiar `Recall@20`, `P@10`
ni las etiquetas de elegibilidad.

## Como Ejecutar la Entrega

Desde la raiz del repositorio:

```powershell
python scripts\run_final_delivery.py --overwrite
```

Ese comando expande el perfil completo P12 aceptado y genera:

```text
results/experiments/FINAL_DELIVERY_TREC2021_P12_FULL_75/
  predictions.json
  trec_metrics.json
  eligibility_metrics.json
  manifest.json
  runner.err / runner.out
```

Para una comprobacion rapida sobre el topic 75, manteniendo TREC eval y
eligibility eval pero sin generar preguntas ni dossiers:

```powershell
python scripts\run_final_delivery.py --topic-ids 75 --metric-only --run-name CHECK_TOPIC_75 --overwrite
```

La documentacion detallada del comando largo equivalente esta en
`entrega_final/COMO_EJECUTAR_ENTREGA.md`.

## Artefactos de Entrega

| Archivo | Contenido |
| --- | --- |
| `entrega_final/predicciones_trec2021_final.json` | Predicciones JSON finales para los 75 topics. |
| `entrega_final/MEMORIA_PROYECTO_FINAL.md` | Memoria tecnica final y analisis. |
| `entrega_final/RESULTADOS_EVALUACION_FINAL.md` | Tabla de metricas, score hackaton y comparativa P11/P12. |
| `entrega_final/metricas_trec_final.json` | TREC con `pytrec_eval`: NDCG@10, Recall@20, MAP, MRR, P@10. |
| `entrega_final/metricas_elegibilidad_final.json` | Micro-F1 y F1 por clase. |
| `entrega_final/metricas_preguntas_final.json` | Evaluacion heuristica T4. |
| `entrega_final/metricas_dossiers_final.json` | Evaluacion heuristica T5. |
| `entrega_final/auditoria_etapas_final.json` | Cobertura de qrels por etapa del agente. |
| `entrega_final/analisis_errores_final.json` | Categorias de error y casos representativos. |
| `entrega_final/manifest_final.json` | Parametros y comandos de la corrida final. |

## Arquitectura del Agente

El pipeline esta implementado como grafo LangGraph con nodos especializados:

```text
perfil paciente
  -> parse_patient
  -> normalize_mesh
  -> plan_search
  -> retrieve_lexical / retrieve_dense
  -> fuse_rrf
  -> rerank_pointwise / rerank_listwise
  -> apply_hard_filters
  -> entity_negation_rerank
  -> select viable candidates
  -> extract criteria
  -> evaluate eligibility
  -> aggregate_to_trial_eval
  -> score_trial
  -> criterion_evidence_adjustment P12
  -> rank
  -> questions
  -> dossiers
  -> JSON final
```

El perfil final de benchmark usa BM25 fielded sobre el snapshot oficial TREC
2021, con `bm25_top_k=1000`, `fused_top_k=1000`, `rerank_top_k=50`,
`max_trials_per_topic=10` y `max_criteria_per_trial=15`.

### Modulos Clave

- **Retrieval estrategico:** BM25 fielded y RRF, con trazas para auditar donde
  se pierden los qrels.
- **Normalizacion clinica:** MeSH y procesamiento estructurado del perfil.
- **Elegibilidad:** extraccion de criterios de inclusion/exclusion y evaluacion
  `met`, `not_met`, `NEI`.
- **Ranking:** score determinista por elegibilidad, fase, estado,
  recencia, geografia y prioridad de retrieval.
- **P11 entity/negation rerank:** reranking benchmark-only con senales de
  condicion, intervencion, negacion y soporte clinico.
- **P12 criterion evidence:** ajuste de ranking por evidencia criterio-a-criterio
  inspirado en TrialGPT, sin usar qrels ni informacion gold.
- **T4 preguntas:** genera preguntas clinicas concretas para resolver criterios
  `NEI`.
- **T5 dossiers:** genera resumen, tabla de elegibilidad, flags y desglose de
  score por ensayo.

## Por Que No Se Usa `diverse_top10` Ni `max_trials_per_topic=20`

El codigo contiene `diverse_top10`, pero no se activa como default porque los
experimentos de 20 topics no demostraron una mejora robusta. Tambien se probo
subir `max_trials_per_topic` a 20: aumento `Recall@20`, pero bajo `NDCG@10`,
`P@10` y `Micro-F1`, ademas de encarecer la corrida. Por eso la entrega final
mantiene `max_trials_per_topic=10` y `benchmark_candidate_selection_policy=top_score`.

## De Donde Salen Las Notas T4 y T5

Las notas de preguntas y dossiers no vienen de qrels oficiales de TREC. Son
metricas internas reproducibles para cubrir T4 y T5 en la formula del reto.

### T4: Calidad de Preguntas

Script: `eval/question_eval.py`.

Modo usado: `rubric`, sin LLM judge.

Cada pregunta se puntua de 0 a 1 a partir de cinco dimensiones, cada una con
0, 1 o 2 puntos:

- `specific`: la pregunta incluye el dato clinico concreto que falta.
- `temporal`: explicita una ventana temporal o contiene una restriccion tipo
  "within".
- `format`: pide un tipo de dato esperado y no solo texto libre.
- `rationale`: incluye razon clinica relacionada con el trial.
- `actionable`: la pregunta tiene forma accionable, longitud razonable y signo
  de interrogacion.

La media final fue `0.913933` sobre 2189 preguntas.

### T5: Completitud de Dossier

Script: `eval/dossier_eval.py`.

Cada dossier se puntua por presencia de campos:

- 70% campos requeridos: `nct_id`, `rank`, `score`, `executive_summary`,
  `metadata`, `eligibility_counts`, `eligibility_table`, `score_breakdown`.
- 20% secciones opcionales: `missing_information`, `attention_flags`,
  `judge_rationale`, `critique_notes`.
- 10% URL de ClinicalTrials.gov en metadata.

La media final fue `0.903993` sobre 1055 dossiers. La URL y el resumen estan
presentes en el 100% de dossiers; la tabla de elegibilidad aparece no vacia en
el 69.95%.

## Instalacion

Requisitos: Python `>=3.11,<3.14`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Configurar credenciales y modelos:

```powershell
copy .env.example .env
```

Editar `.env` con las claves reales. No subir `.env` al repositorio.

Configuracion documentada en `.env.example`:

- Proveedor principal: Azure OpenAI.
- Deployment mini: `gpt-5-mini`.
- Deployment large: `gpt-5.4`.
- Temperatura de extraccion: `0.0`.
- Temperatura judge: `0.2`.
- Temperatura self-consistency: `0.5`.
- Reasoning effort estructurado: `minimal`.

## Datos e Indices

La entrega final asume que existen:

```text
data/trec_ct/raw/topics_2021.xml
data/trec_ct/raw/qrels_2021.txt
data/trec_ct/clinical_trials_2021_04_27
data/indices/bm25_trec2021_fielded
```

Scripts utiles:

```powershell
bash scripts/download_trec.sh
python scripts\download_trec_ct_snapshot.py
python scripts\build_fielded_bm25_index.py --ctgov-dir data\trec_ct\clinical_trials_2021_04_27 --output-dir data\indices\bm25_trec2021_fielded --input-format ctgov-legacy-xml --no-filter --source "TREC Clinical Trials 2021" --snapshot-date 2021-04-27 --corpus-manifest-out data\corpus\trec2021_snapshot_manifest.json --write-index-manifest
```

En Windows, si se usan los `.sh`, ejecutarlos desde Git Bash o WSL.

## Evaluacion Manual

```powershell
python eval\trec_eval.py --predictions entrega_final\predicciones_trec2021_final.json --qrels data\trec_ct\raw\qrels_2021.txt --output results\trec_metrics_check.json
python eval\eligibility_eval.py --predictions entrega_final\predicciones_trec2021_final.json --qrels data\trec_ct\raw\qrels_2021.txt --output results\eligibility_metrics_check.json
python eval\question_eval.py --predictions entrega_final\predicciones_trec2021_final.json --output results\question_metrics_check.json
python eval\dossier_eval.py --predictions entrega_final\predicciones_trec2021_final.json --output results\dossier_metrics_check.json
```

Validacion tecnica:

```powershell
python -m pytest tests\unit -q
python -m compileall src eval scripts tests
```

Ultima validacion local: `177 passed` y `compileall` correcto.

## Comparativa y Analisis de Errores

Baseline principal: P11, con BM25 fielded, few-shot, multisignal class-0 y
entity/negation rerank.

| Run | NDCG@10 | Recall@20 | P@10 | MAP | MRR | Micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P11 baseline 75 | 0.517134 | 0.094129 | 0.630667 | 0.080269 | 0.744704 | 0.499281 |
| P12 final 75 | 0.520406 | 0.094129 | 0.630667 | 0.080258 | 0.763815 | 0.499281 |

El mayor cuello sigue siendo el paso de candidatos finales a trials evaluados:
BM25@1000 recupera mucho oro, pero el ranking final pierde candidatos relevantes
por falsos positivos de retrieval, hard-vetoes y evidencia incompleta.

## Originalidad

La parte mas diferencial no es solo usar RAG: el sistema separa planificacion,
retrieval, filtros, elegibilidad criterio-a-criterio, ranking, preguntas y
dossier. P12 anade una capa auditable de evidencia por criterio que explica por
que un ensayo sube o baja sin mirar qrels.

## Limitaciones

- No implementa monitorizacion continua T6; era opcional.
- La clase 0 tiene recall bajo: detecta pocos irrelevantes, aunque con alta
  precision.
- `Recall@20` final queda limitado por selection loss entre retrieval y ranking.
- Las metricas T4/T5 son heuristicas internas, no juicios clinicos humanos.

## Estructura del Repositorio

```text
src/trial_matcher/        codigo del agente
scripts/                  wrappers y utilidades reproducibles
eval/                     evaluadores TREC, elegibilidad, preguntas y dossiers
tests/                    tests unitarios e integracion
data/                     TREC, corpus e indices locales
entrega_final/            artefactos finales para entregar
memoria/                  memoria tecnica editable
```

