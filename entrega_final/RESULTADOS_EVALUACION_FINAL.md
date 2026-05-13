# Resultados de Evaluacion Final

Fecha de cierre: 2026-05-13  
Run final: `results/experiments/FINAL_DELIVERY_TREC2021_P12_FULL_75`  
Predicciones finales: `entrega_final/predicciones_trec2021_final.json`

## Configuracion aceptada

La configuracion final usa el baseline P11 aceptado y anade P12 como ajuste de ranking por evidencia a nivel de criterio:

- Corpus: TREC Clinical Trials 2021, 75 topics.
- Retrieval: BM25 fielded, indice `data/indices/bm25_trec2021_fielded`.
- Candidatos: `bm25_top_k=1000`, `fused_top_k=1000`, `rerank_top_k=50`.
- Eligibility: `max_trials_per_topic=10`, `max_criteria_per_trial=15`, few-shot activado.
- Politicas benchmark: multisignal class-0 activada, entity/negation rerank activado con peso `0.09` y proteccion top `3`.
- P12: `benchmark_criterion_evidence_policy=score_adjust`, peso `0.50`.
- T4/T5: preguntas y dossiers activados en la corrida final.
- Para preservar la comparabilidad del ranking, se mantuvieron desactivados `self_consistency`, `llm_judge` y `self_critique`.

## Metrica TREC oficial con pytrec_eval

| Run | NDCG@10 | Recall@20 | P@10 | MAP | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| P11 baseline 75 | 0.517134 | 0.094129 | 0.630667 | 0.080269 | 0.744704 |
| P12 final 75 | 0.520406 | 0.094129 | 0.630667 | 0.080258 | 0.763815 |
| Delta P12 - P11 | +0.003271 | +0.000000 | +0.000000 | -0.000011 | +0.019111 |

El evaluador final es `pytrec_eval`, instalado y validado tras la corrida. Tambien se conserva el fallback en `entrega_final/metricas_trec_pure_python_final.json` para trazabilidad.

## Score Hackaton

Formula del reto:

```text
Score = 0.20 * Recall@20 + 0.30 * Micro-F1 + 0.25 * NDCG@10 + 0.15 * calidad preguntas + 0.10 * completitud dosier
```

Con la corrida final P12 completa:

| Componente | Valor | Peso | Aporte |
| --- | ---: | ---: | ---: |
| Recall@20 | 0.094129 | 0.20 | 0.018826 |
| Micro-F1 | 0.499281 | 0.30 | 0.149784 |
| NDCG@10 | 0.520406 | 0.25 | 0.130101 |
| Calidad preguntas | 0.913933 | 0.15 | 0.137090 |
| Completitud dosier | 0.903993 | 0.10 | 0.090399 |
| **Score total** |  |  | **0.526201** |

Equivalente aproximado: **52.62 / 100** si se expresa como porcentaje.

## Eligibility

| Metrica | Valor |
| --- | ---: |
| Micro-F1 | 0.499281 |
| Accuracy | 0.499281 |
| Macro-F1 | 0.340322 |
| Class-2 precision | 0.747664 |
| Class-2 recall | 0.265781 |
| Class-2 F1 | 0.392157 |
| Class-0 precision | 1.000000 |
| Class-0 recall | 0.009091 |
| Class-0 F1 | 0.018018 |

P12 no cambia etiquetas de elegibilidad, solo reordena por score. Por eso Micro-F1 y Class-2 F1 se mantienen exactamente igual que P11, mientras mejora el ranking temprano.

## Preguntas y Dossiers

| Salida | Resultado |
| --- | ---: |
| Preguntas generadas | 2189 |
| Rubric score medio de preguntas | 0.913933 |
| Dossiers generados | 1055 |
| Completitud media de dossiers | 0.903993 |
| Dossiers con URL CT.gov | 1.000000 |
| Dossiers con resumen | 1.000000 |
| Dossiers con tabla de elegibilidad no vacia | 0.699526 |

## Auditoria de Etapas

Puntos clave de `auditoria_etapas_final.json`:

- `bm25_candidates@1000` recupera recall medio relevante `0.675390` y eligible `0.680705`.
- `final_candidates@20` tiene recall medio relevante `0.126394` y eligible `0.135235`.
- `ranked_trials@20` baja a recall relevante `0.094129` y eligible `0.101002`.
- Gap `final_candidates@20 -> ranked_trials@20`: 535 candidatos perdidos, 251 relevantes perdidos y 147 elegibles perdidos.
- Topics sin relevante en ranking: `15`, `31`.
- Topics sin eligible en ranking: `9`, `15`, `23`, `27`, `31`, `34`, `44`, `61`, `70`.

## Analisis de Error

Categorias principales en `analisis_errores_final.json`:

- `retrieval_false_positive`: 463.
- `retrieval_false_positive_unjudged`: 354.
- `missed_gold_eligible`: 221.
- `hard_veto_gold_eligible`: 129.
- `low_inclusion_support_gold_eligible`: 180.
- `high_nei_gold_eligible`: 103.
- `exclusion_met_gold_eligible`: 71.
- `overcalled_eligible`: 55.

Lectura critica: P12 mejora la posicion de candidatos mejor soportados, pero el cuello estructural no desaparece. La mayor deuda sigue siendo el paso entre `final_candidates` y `ranked_trials`, especialmente por falsos positivos de retrieval y por oro eligible que llega al pool pero no sobrevive a evaluacion/cap/score.

## Artefactos Entregados

- `predicciones_trec2021_final.json`: fichero JSON final de predicciones no interactivas.
- `metricas_trec_final.json`: metricas TREC recalculadas con `pytrec_eval`.
- `metricas_trec_pure_python_final.json`: metrica TREC con fallback historico.
- `metricas_elegibilidad_final.json`: F1/precision/recall por clase.
- `auditoria_etapas_final.json`: auditoria por etapa y gap de candidatos.
- `analisis_errores_final.json`: categorias de error y casos representativos.
- `metricas_preguntas_final.json`: evaluacion heuristica de preguntas.
- `metricas_dossiers_final.json`: completitud de dossiers.
- `manifest_final.json`: comandos y parametros de la corrida final.
- `MEMORIA_PROYECTO_FINAL.md`: memoria tecnica del proyecto.
- `COMO_EJECUTAR_ENTREGA.md`: instrucciones reproducibles.
