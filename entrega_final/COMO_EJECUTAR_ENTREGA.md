# Como Ejecutar la Entrega

## Ejecucion no interactiva final

Desde la raiz del proyecto:

```powershell
python scripts\run_final_delivery.py --overwrite
```

El comando procesa los 75 topics de `data/trec_ct/raw/topics_2021.xml`, no requiere interaccion humana y produce `predictions.json`, `trec_metrics.json`, `eligibility_metrics.json`, `manifest.json` y logs en:

```text
results/experiments/FINAL_DELIVERY_TREC2021_P12_FULL_75
```

La version copiada para entrega es:

```text
entrega_final/predicciones_trec2021_final.json
```

Para una comprobacion rapida sobre el topic 75, sin generar preguntas ni dossiers:

```powershell
python scripts\run_final_delivery.py --topic-ids 75 --metric-only --run-name CHECK_TOPIC_75 --overwrite
```

`--metric-only` solo desactiva T4/T5 (`questions` y `dossiers`); la prediccion, TREC eval y eligibility eval se siguen ejecutando.

## Comando equivalente auditable

El wrapper anterior expande exactamente este perfil P12 aceptado:

```powershell
python scripts\run_mini_eval.py --year 2021 --topic-limit 75 --top-k 20 --ctgov-dir data\trec_ct\clinical_trials_2021_04_27 --bm25-mode fielded --fielded-bm25-index-dir data\indices\bm25_trec2021_fielded --bm25-top-k 1000 --no-dense --fused-top-k 1000 --rerank-top-k 50 --require-benchmark-index-manifest --concurrency 4 --mode benchmark --max-trials-per-topic 10 --max-criteria-per-trial 15 --run-name FINAL_DELIVERY_TREC2021_P12_FULL_75 --overwrite --enable-few-shot --no-self-consistency --no-llm-judge --no-self-critique --include-retrieval-traces --benchmark-min-inclusion-fraction 0.1 --benchmark-max-nei-fraction 1.0 --enable-multisignal-irrel-heuristic --benchmark-candidate-selection-policy top_score --benchmark-entity-rerank-policy rerank_final --benchmark-entity-rerank-weight 0.09 --benchmark-entity-protect-top 3 --benchmark-criterion-evidence-policy score_adjust --benchmark-criterion-evidence-weight 0.50
```

La configuracion final mantiene `max_trials_per_topic=10` y `benchmark_candidate_selection_policy=top_score`. No activa `diverse_top10` como default, y no sube el cap a 20 porque en los experimentos previos aumento `Recall@20` pero bajo `NDCG@10`, `P@10` y `Micro-F1`.

## Evaluaciones adicionales

```powershell
python eval\trec_eval.py --predictions results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\predictions.json --qrels data\trec_ct\raw\qrels_2021.txt --output results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\trec_metrics_pytrec_eval.json
python eval\eligibility_eval.py --predictions results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\predictions.json --qrels data\trec_ct\raw\qrels_2021.txt --output results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\eligibility_metrics.json
python eval\agent_stage_audit.py --predictions results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\predictions.json --qrels data\trec_ct\raw\qrels_2021.txt --output results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\stage_audit.json
python eval\error_analysis.py --predictions results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\predictions.json --qrels data\trec_ct\raw\qrels_2021.txt --out results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\error_analysis.json
python eval\question_eval.py --predictions results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\predictions.json --output results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\question_metrics.json
python eval\dossier_eval.py --predictions results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\predictions.json --output results\experiments\FINAL_DELIVERY_TREC2021_P12_FULL_75\dossier_metrics.json
```

## Validacion tecnica

```powershell
python -m pytest tests\unit -q
python -m compileall src eval scripts tests
```

Resultado verificado: `177 passed` y `compileall` correcto.
