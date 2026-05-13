# Trial Matcher - Memoria tecnica

Fecha: 2026-05-13  
Entrega: `entrega_final/predicciones_trec2021_final.json`  
Benchmark: TREC Clinical Trials 2021, 75 topics

## 1. Resumen ejecutivo

Trial Matcher es un agente no interactivo para matching paciente-ensayo clinico.
El sistema recibe perfiles de paciente, recupera ensayos de ClinicalTrials.gov,
evalua criterios de inclusion/exclusion, ordena candidatos, genera preguntas de
informacion faltante y produce dossiers estructurados para revision medica.

El resultado final corresponde a la version P12. La mejora P12 introduce un
ajuste de ranking por evidencia a nivel de criterio: no cambia las etiquetas de
elegibilidad, pero usa la tabla de criterios ya evaluada para subir ensayos con
soporte clinico sustantivo y bajar candidatos con evidencia debil, exclusiones
o incertidumbre critica. La politica es benchmark-only y no usa qrels ni datos
gold durante inferencia.

Score final del reto:

```text
0.20 * Recall@20      = 0.018826
0.30 * Micro-F1       = 0.149784
0.25 * NDCG@10        = 0.130101
0.15 * preguntas      = 0.137090
0.10 * dossier        = 0.090399
Score total           = 0.526201
```

Equivalente: 52.62 / 100.

## 2. Arquitectura del agente

El agente esta implementado como un grafo LangGraph con nodos separados por
responsabilidad:

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
  -> candidate selection
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

La configuracion final usa BM25 fielded sobre el snapshot TREC 2021 oficial,
con `bm25_top_k=1000`, `fused_top_k=1000`, `rerank_top_k=50`,
`max_trials_per_topic=10` y `max_criteria_per_trial=15`. Dense retrieval,
HyDE, listwise, verifier, self-consistency, LLM judge y self-critique se
desactivan en la corrida final para preservar la comparabilidad del baseline
validado y reducir coste.

Decisiones clave:

- Retrieval amplio al inicio, pero cap de elegibilidad controlado para no
  saturar el LLM.
- Evaluacion criterio-a-criterio con labels `met`, `not_met`, `NEI`.
- Agregacion a clases TREC: `irrelevant` (0), `excludes` (1), `eligible` (2).
- Entity/negation rerank P11 para mover candidatos usando senales medicas
  simples antes del cap de elegibilidad.
- Criterion evidence P12 para ordenar mejor sin tocar etiquetas.
- Preguntas y dossiers generados como salidas clinicamente utiles, no solo
  como metrica.

## 3. Configuracion de modelos e inferencia

La configuracion documentada esta en `.env.example`.

- Proveedor principal: Azure OpenAI.
- Deployment mini: `gpt-5-mini`.
- Deployment large: `gpt-5.4`.
- Temperatura de extraccion: `0.0`.
- Temperatura judge: `0.2`.
- Temperatura self-consistency: `0.5`.
- Reasoning effort estructurado: `minimal`.
- Few-shot dinamico: activado.
- Self-consistency, verifier, LLM judge y self-critique: desactivados en la
  corrida final de entrega.

La ejecucion final se hace mediante:

```powershell
python scripts\run_final_delivery.py --overwrite
```

El comando completo equivalente queda registrado en
`entrega_final/COMO_EJECUTAR_ENTREGA.md` y en `entrega_final/manifest_final.json`.

## 4. Resultados

Metricas TREC finales con `pytrec_eval`:

| Metrica | Valor |
| --- | ---: |
| NDCG@10 | 0.520406 |
| Recall@20 | 0.094129 |
| P@10 | 0.630667 |
| MAP | 0.080258 |
| MRR | 0.763815 |

Elegibilidad:

| Metrica | Valor |
| --- | ---: |
| Micro-F1 | 0.499281 |
| Class-2 F1 | 0.392157 |
| Class-1 F1 | 0.610792 |
| Class-0 F1 | 0.018018 |
| Macro-F1 | 0.340322 |

Preguntas y dossiers:

| Salida | Valor |
| --- | ---: |
| Preguntas generadas | 2189 |
| Calidad media preguntas | 0.913933 |
| Dossiers generados | 1055 |
| Completitud media dossier | 0.903993 |

Comparativa principal:

| Run | NDCG@10 | Recall@20 | P@10 | MAP | MRR | Micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P11 baseline 75 | 0.517134 | 0.094129 | 0.630667 | 0.080269 | 0.744704 | 0.499281 |
| P12 final 75 | 0.520406 | 0.094129 | 0.630667 | 0.080258 | 0.763815 | 0.499281 |

P12 se acepta porque mejora ranking temprano (`NDCG@10` y `MRR`) sin degradar
la parte de elegibilidad. El cambio en MAP es despreciable.

## 5. De donde salen las notas de preguntas y dossier

TREC no proporciona qrels oficiales para T4 y T5, asi que se usan evaluadores
heuristicos internos, reproducibles y sin LLM judge.

### T4: calidad de preguntas

Script: `eval/question_eval.py`.

Cada pregunta se puntua con cinco dimensiones, cada una en escala 0, 1 o 2. La
nota se normaliza dividiendo entre 10.

- `specific`: contiene el dato clinico exacto que falta.
- `temporal`: incluye ventana temporal o restriccion de tiempo.
- `format`: pide un tipo de dato esperado, no solo texto libre.
- `rationale`: explica por que el dato importa para el trial.
- `actionable`: es una pregunta accionable, con longitud razonable.

Resultado final: 2189 preguntas, media `0.913933`.

### T5: completitud de dossier

Script: `eval/dossier_eval.py`.

La completitud combina:

- 70% campos requeridos presentes: `nct_id`, `rank`, `score`,
  `executive_summary`, `metadata`, `eligibility_counts`,
  `eligibility_table`, `score_breakdown`.
- 20% secciones opcionales presentes: `missing_information`,
  `attention_flags`, `judge_rationale`, `critique_notes`.
- 10% URL de ClinicalTrials.gov presente.

Resultado final: 1055 dossiers, media `0.903993`. El 100% tiene URL y resumen;
el 69.95% tiene tabla de elegibilidad no vacia.

Estas notas deben presentarse como metricas internas de calidad/completitud, no
como juicios clinicos humanos.

## 6. Analisis de errores

La auditoria muestra que el corpus y el indice no son el cuello principal:
`bm25_candidates@1000` alcanza recall relevante medio `0.675390`. El problema
aparece despues, al pasar de candidatos amplios a ensayos evaluados y rankeados.

Puntos principales:

- `final_candidates@20`: recall relevante medio `0.126394`.
- `ranked_trials@20`: recall relevante medio `0.094129`.
- Gap `final_candidates@20 -> ranked_trials@20`: 251 relevantes perdidos y 147
  eligible perdidos.
- Errores frecuentes: falsos positivos de retrieval, hard-veto sobre gold
  eligible, bajo soporte de inclusion y NEI alto.

La clase 0 sigue siendo la mas debil: tiene precision alta en pocas
activaciones, pero recall muy bajo. Es una decision conservadora: preferimos no
convertir qrel=1/qrel=2 en qrel=0 salvo cuando hay varias senales de
irrelevancia semantica.

## 7. Originalidad

El sistema no es un RAG simple. La contribucion esta en la composicion de
herramientas y decisiones:

- Planificacion separada del retrieval.
- Evaluacion criterio-a-criterio con salida estructurada.
- Auditoria de trazas por etapa para saber donde se pierde el oro.
- Entity/negation rerank sin qrels.
- P12 criterion evidence: ranking interpretable por evidencia de criterios.
- Generacion de preguntas y dossiers conectados con los criterios `NEI`.

## 8. Reproducibilidad

Comando principal:

```powershell
python scripts\run_final_delivery.py --overwrite
```

Comprobacion rapida de topic 75:

```powershell
python scripts\run_final_delivery.py --topic-ids 75 --metric-only --run-name CHECK_TOPIC_75 --overwrite
```

Validacion tecnica local:

```powershell
python -m pytest tests\unit -q
python -m compileall src eval scripts tests
```

Ultimo resultado: `177 passed` y `compileall` correcto.

## 9. Limitaciones y roadmap

Limitaciones:

- No se implementa monitorizacion continua T6; era opcional.
- `Recall@20` queda limitado por selection loss.
- Clase 0 con recall bajo.
- T4/T5 se evaluan con rubricas heuristicas, no con revision humana.

Roadmap:

- Reranking semantico mas fuerte antes de eligibility.
- Mejor politica de clase 0 por condicion, poblacion e intervencion.
- Calibracion de hard-veto para qrels TREC.
- Integracion FHIR/EHR para perfiles reales.
- Monitorizacion continua de cambios en ClinicalTrials.gov.

## 10. Conclusiones

La entrega cumple los requisitos principales: agente ejecutable de forma no
interactiva, predicciones JSON, metricas TREC, elegibilidad criterio-a-criterio,
preguntas de informacion faltante, dossiers y analisis de errores. P12 aporta
una mejora pequena pero real de ranking y deja una explicacion auditable de por
que cambia el orden de los ensayos.
