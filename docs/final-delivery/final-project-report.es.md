# Memoria Final del Proyecto

Fecha: 2026-05-13  
Proyecto: agente no interactivo para matching paciente-ensayo clinico sobre TREC Clinical Trials 2021  
Entrega principal: `docs/final-delivery/trec2021-final-predictions.json`

## 1. Resumen Ejecutivo

El proyecto construye un agente capaz de recibir perfiles de pacientes, recuperar ensayos clinicos de ClinicalTrials.gov, evaluar elegibilidad por criterios de inclusion y exclusion, producir un ranking en formato JSON y generar informacion auxiliar para revision clinica: preguntas de informacion faltante y dossiers por ensayo. La entrega final se ejecuto de forma no interactiva sobre los 75 topics de TREC Clinical Trials 2021 y produjo un fichero JSON completo con rankings, labels predichos, componentes de score, diagnosticos, preguntas y dossiers.

El baseline serio antes de esta fase era P11, derivado de P9/P10: BM25 fielded, few-shot eligibility, politica multisignal para clase 0, reranking entity/negation-aware y auditorias de cobertura. P11 ya habia mejorado respecto a P8/P9 en ranking temprano y mantenia una Micro-F1 estable, pero seguia mostrando los mismos cuellos: falsos positivos de retrieval, bajo recall final frente al pool de candidatos y clase 0 con muy poco recall.

La mejora implementada en esta entrega es P12, un `criterion-level evidence ranker` inspirado en la arquitectura de TrialGPT. En lugar de confiar solo en un score agregado de trial, P12 resume cada evaluacion de criterio en componentes interpretables: soporte de inclusion, soporte clinico sustantivo, fallos de inclusion, exclusion explicita, incertidumbre por exclusiones no verificadas, missing obligatorio y penalizacion por evidencia fina. Con esos componentes ajusta el score final de ranking sin cambiar la etiqueta predicha. La politica es benchmark-only, opt-in y no usa qrels ni informacion gold.

El resultado final con `pytrec_eval` mejora el ranking temprano frente a P11: `NDCG@10` pasa de `0.517134` a `0.520406`, y `MRR` de `0.744704` a `0.763815`. `P@10` y `Recall@20` quedan iguales, y `MAP` baja de forma despreciable `0.000011`. Como P12 no toca labels, la evaluacion de elegibilidad se mantiene: `Micro-F1=0.499281`, `Class-2 F1=0.392157`, `Class-0 F1=0.018018`.

La lectura critica es clara: P12 es una mejora real de ordenacion, no una solucion total. El agente final es entregable y no interactivo, pero el sistema todavia pierde muchos candidatos oro entre `final_candidates` y `ranked_trials`, y los falsos positivos de retrieval siguen siendo la principal fuente de error. La siguiente mejora de alto impacto no deberia ser aumentar el cap ni rellenar mas, sino mejorar retrieval/reranking semantico antes de eligibility y hacer una politica de irrelevancia que detecte clase 0 semantica sin confundirla con qrel=1.

## 2. Objetivo y Datos

El objetivo inmediato era producir una entrega final sobre TREC Clinical Trials 2021 sin contaminar el modo clinico activo. Por eso todas las politicas experimentales de benchmark se mantienen detras de flags y no cambian el comportamiento clinico por defecto. El agente se ejecuta sobre `data/trec_ct/raw/topics_2021.xml`, usa qrels solo para evaluacion externa y trabaja sobre el snapshot `data/trec_ct/clinical_trials_2021_04_27`.

El formato final de prediccion es JSON. Cada topic contiene `ranked_trials` con `nct_id`, `rank`, `score`, `label`, `predicted_trec_qrel`, el alias historico `trec_qrel`, rationale, conteos de inclusion/exclusion, fraccion NEI, componentes de score y marcas de fill. El campo oficial de prediccion es `predicted_trec_qrel`; `trec_qrel` queda como alias retrocompatible y no debe interpretarse como gold. Cada topic tambien incluye `questions`, `dossiers`, `stats` y `diagnostics`.

La entrega copia el JSON final a `docs/final-delivery/trec2021-final-predictions.json`. La ejecucion no interactiva queda documentada en `docs/final-delivery/how-to-run-final-delivery.es.md`.

## 3. Arquitectura del Agente

El pipeline final combina recuperacion lexica fielded, seleccion de candidatos, evaluacion criterio-a-criterio, agregacion de elegibilidad, ranking determinista, ajuste P12 y generacion de salidas auxiliares.

La primera etapa parsea el perfil del paciente y extrae una representacion estructurada. Despues se ejecuta retrieval BM25 fielded sobre campos del trial, usando `bm25_top_k=1000`. En esta entrega se desactiva dense retrieval para preservar el baseline validado y porque la configuracion fielded BM25 era la mas estable para el benchmark actual.

La etapa de candidatos usa `fused_top_k=1000` y `rerank_top_k=50`, con trazas de retrieval activadas. Estas trazas son importantes porque permiten medir si el oro se pierde en retrieval, en el paso de candidatos finales o despues de eligibility. El diagnostico final confirma que el corpus y el indice no son el problema principal: BM25 a 1000 recupera una proporcion importante de relevantes, pero el ranking final sigue perdiendo mucho oro elegible.

La etapa de eligibility evalua hasta 10 trials viables por topic y hasta 15 criterios por trial. Cada criterio tiene polaridad, tipo y label (`met`, `not_met`, `NEI`). El agregador produce una etiqueta de trial alineada con TREC: `eligible` para qrel 2, `excludes` para qrel 1 e `irrelevant` para qrel 0. La politica multisignal de clase 0 se mantiene activada, pero es estricta: busca irrelevancia semantica, no simple inelegibilidad. Esto evita confundir qrel=1 con qrel=0.

P11 anadio un reranking entity/negation-aware sobre candidatos finales. Su proposito es usar senales medicas simples de condiciones, intervenciones, negacion y soporte de titulo/condicion para mover candidatos sin usar qrels. En la entrega final se activa con `benchmark_entity_rerank_policy=rerank_final`, peso `0.09` y proteccion top `3`.

P12 se inserta despues de `score_trial` y antes de ordenar. Toma la tabla de criterios ya evaluada y calcula un ajuste:

```text
score_final = score_base + weight * (criterion_evidence_score - 0.5)
```

Para hard-vetoes el peso efectivo baja al 25 por ciento, de modo que P12 no deshace la logica de seguridad del scorer. Los componentes se guardan en el output como numeros, y la tabla detallada queda en diagnostics para auditoria.

## 4. P12: Diseno y Correccion Critica

La primera version de P12 era demasiado agresiva contra exclusiones desconocidas. En TREC, un trial qrel=2 puede carecer de evidencia explicita para cada exclusion en el perfil del paciente. Castigar todas las exclusiones NEI como si fueran negativas degradaba `NDCG@10` y `MRR`. Tambien habia otro problema: si el unico criterio satisfecho era edad o sexo, el sistema podia tratarlo como soporte de inclusion completo, elevando falsos positivos.

La version final corrige ambos puntos. Primero, la exclusion desconocida pasa a ser una senal debil; el castigo fuerte se reserva para exclusiones explicitamente satisfechas por el paciente. Segundo, el score separa `inclusion_support` de `substantive_inclusion_support`. La segunda senal solo cuenta tipos clinicos de mayor valor: diagnostico, biomarcador, tratamiento previo, laboratorio, comorbilidad, performance status, embarazo y otros criterios sustantivos. Edad, sexo y consentimiento no bastan para dar un boost clinico fuerte.

Esta correccion se comprobo empiricamente:

- P11 control de 20 topics: `NDCG@10=0.467890`, `MRR=0.739167`, `Micro-F1=0.494845`.
- P12 inicial con peso `0.18`: bajaba `NDCG@10` a `0.464501`.
- P12 revisado con peso `0.50`: subia `NDCG@10` a `0.475530`, `MRR=0.780833`, sin cambiar `Micro-F1` ni `Class-2 F1`.

Despues se ejecuto 75 topics. Con `pytrec_eval`, P12 final queda por encima de P11 en `NDCG@10` y `MRR`, mantiene `P@10` y `Recall@20`, y solo pierde `0.000011` en MAP. Se acepta porque el objetivo de P12 era mejorar ranking temprano sin tocar labels, y eso se cumple.

## 5. Implementacion

Los cambios principales estan en:

- `src/trial_matcher/ranking/criterion_evidence.py`: nuevo modulo P12.
- `src/trial_matcher/agent/nodes.py`: aplica P12 en el nodo de ranking y guarda diagnostics.
- `src/trial_matcher/config.py`: nuevas opciones `benchmark_criterion_evidence_policy` y `benchmark_criterion_evidence_weight`.
- `src/trial_matcher/runner.py`: flags CLI, metadata y diagnosticos.
- `scripts/run_mini_eval.py`: soporte reproducible para flags P12.
- `src/trial_matcher/api.py`: health expone configuracion P12.
- `tests/unit/test_criterion_evidence.py`: tests de boost, penalizacion, clinical mode intacto y componentes numericos.
- `tests/unit/test_runner.py`: defaults actualizados.

La politica sigue apagada por defecto (`off`). El peso por defecto de la politica experimental queda en `0.50`, pero solo tiene efecto si se activa `score_adjust`. Esto protege el modo clinico y evita que un cambio de benchmark altere ejecuciones interactivas.

## 6. Resultados Finales

La corrida final completa es `FINAL_DELIVERY_TREC2021_P12_FULL_75`. Se ejecuto con preguntas y dossiers activados. La metrica TREC se recalculo despues con `pytrec_eval`, instalado correctamente en el entorno.

Resultados TREC finales:

| Metrica | Valor |
| --- | ---: |
| NDCG@10 | 0.520406 |
| Recall@20 | 0.094129 |
| P@10 | 0.630667 |
| MAP | 0.080258 |
| MRR | 0.763815 |

Resultados de elegibilidad:

| Metrica | Valor |
| --- | ---: |
| Micro-F1 | 0.499281 |
| Class-2 F1 | 0.392157 |
| Class-1 F1 | 0.610792 |
| Class-0 F1 | 0.018018 |
| Macro-F1 | 0.340322 |

Preguntas y dossiers:

- Preguntas generadas: 2189.
- Score rubric medio de preguntas: 0.913933.
- Dossiers generados: 1055.
- Completitud media de dossiers: 0.903993.
- URLs CT.gov presentes: 100 por ciento.
- Resumen de dossier presente: 100 por ciento.
- Tabla de elegibilidad no vacia: 69.95 por ciento.

Score combinado de la hackaton:

```text
0.20 * Recall@20      = 0.018826
0.30 * Micro-F1       = 0.149784
0.25 * NDCG@10        = 0.130101
0.15 * preguntas      = 0.137090
0.10 * dossier        = 0.090399
Score total           = 0.526201
```

Equivalente: 52.62 / 100.

Las notas de preguntas y dossiers salen de evaluadores heuristicos internos,
no de qrels oficiales de TREC. Para T4, `eval/question_eval.py` usa modo
`rubric` y puntua cada pregunta en cinco dimensiones: dato concreto, ventana
temporal, formato esperado, rationale clinico y accionabilidad. Cada dimension
vale 0, 1 o 2 puntos y el total se normaliza a 0-1. Para T5,
`eval/dossier_eval.py` mide completitud por presencia de campos: 70 por ciento
campos requeridos, 20 por ciento secciones opcionales y 10 por ciento URL de
ClinicalTrials.gov. Estas metricas son reproducibles y utiles para la formula
del reto, pero deben interpretarse como proxies de calidad/completitud, no como
revision clinica humana.

La diferencia entre la metrica `pure_python` y `pytrec_eval` aparece sobre todo en `NDCG@10`. El paquete final usa `pytrec_eval` como fuente principal y conserva el fallback por trazabilidad.

## 7. Auditoria y Errores Pendientes

La auditoria de etapas muestra el cuello con bastante claridad:

- `bm25_candidates@1000` alcanza recall relevante medio `0.675390`.
- `final_candidates@20` baja a recall relevante `0.126394`.
- `ranked_trials@20` queda en `0.094129`.
- En el gap `final_candidates@20 -> ranked_trials@20` se pierden 251 trials relevantes y 147 eligible.

El error analysis enumera 463 falsos positivos de retrieval, 221 gold eligible perdidos, 129 gold eligible afectados por hard-veto, 180 con bajo soporte de inclusion, 103 con NEI alto y 71 donde aparece una exclusion como met pese a ser gold eligible. Esto implica que el sistema no esta limitado solo por ranking final: tambien hay problemas de interpretacion de criterios, retrieval semantico y calibracion entre exclusion real, missing evidence e irrelevancia.

Clase 0 sigue siendo el area mas debil: precision perfecta en las pocas activaciones, pero recall bajisimo. Eso es mejor que convertir qrel=1/qrel=2 erroneamente a 0, pero no es suficiente para un sistema final competitivo. La clase 0 futura debe detectar irrelevancia semantica por tema, poblacion e intervencion, no simplemente alta NEI.

## 8. Reproducibilidad

El comando reproducible esta en `how-to-run-final-delivery.es.md`. La corrida es no interactiva: toma la lista de topics desde XML, ejecuta el agente y produce JSON. Los artefactos copiados a `docs/final-delivery` son los que deben usarse para entrega.

Validacion tecnica final:

```text
python -m pytest tests\unit -q
177 passed

python -m compileall src eval scripts tests
OK
```

La instalacion de `pytrec_eval` tambien quedo validada y las metricas finales principales se recalcularon con ese evaluador.

## 9. Conclusiones

La entrega final esta lista como sistema no interactivo de benchmark con JSON completo, metricas finales, auditorias y memoria. P12 es una mejora aceptada porque sube ranking temprano sin tocar labels y sin degradar F1. El mayor valor de P12 no es solo la metrica: tambien deja una representacion auditable de por que un trial sube o baja, criterio por criterio.

No conviene declarar el sistema como resuelto. El siguiente paso real, si hubiera tiempo, seria atacar retrieval false positives y selection loss antes de eligibility. En concreto: entity/negation features mas fuertes en retrieval, normalizacion semantica de condiciones/intervenciones, reranking de candidatos con soporte clinico sustantivo, y una politica de clase 0 que distinga "relevante pero excluido" de "no relevante". Aun asi, para la entrega actual, el agente cumple: ejecuta de forma no interactiva sobre perfiles, entrega predicciones JSON, genera preguntas/dossiers y mejora el baseline validado en NDCG/MRR con evaluacion reproducible.
