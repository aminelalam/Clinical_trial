# Auditoría Profunda y Crítica del Código: Trial Matcher

Este documento representa un escrutinio absoluto del código fuente del proyecto Trial Matcher. A diferencia de las guías de alto nivel, aquí bajaremos al barro: analizaremos las clases de Python, las estructuras de Pydantic, los campos exactos, los tipos de datos, los nodos de LangGraph y los algoritmos matemáticos con mentalidad de ingeniero de software sénior.

---

## 1. Estructura de Datos Central: El Contrato `AgentState`

El núcleo de todo el sistema se encuentra en `src/trial_matcher/models/agent_state.py`.
LangGraph requiere un estado que se actualice mediante reductores. Para ello, el proyecto implementa un diseño brillante pero complejo: la dualidad entre `AgentStateDict` (TypedDict para el runtime de LangGraph) y `AgentState` (Pydantic para validación y testing).

### Campos Críticos del Estado:
1. **Entradas y Salidas:**
   - `patient_raw` (`str`): El texto XML/texto libre original que entra al sistema.
   - `patient_profile` (`PatientProfile`): Objeto Pydantic instanciado por el LLM. Contiene campos clave: `demographics` (edad, sexo), `diagnoses` (condiciones), `biomarkers`, `prior_treatments`. Este objeto dirige las "Hard Filters".
   - `search_plan` (`SearchPlan`): Objeto que contiene listas de queries como `must_include`, `should_include`.

2. **Listas de Candidatos en Memoria:**
   - `bm25_candidates`, `fused_candidates`, `final_candidates`: Son listas de objetos `TrialCandidate`. 
   - Un `TrialCandidate` es extremadamente ligero: `nct_id`, `score`, `rank`, `hard_excluded` (booleano), `excluded_reason`.
   - *Crítica Arquitectónica:* Es un gran acierto de diseño no guardar todo el texto del ensayo en el estado del grafo, lo que provocaría un desbordamiento de memoria (OOM). El estado viaja ligero con IDs y puntuaciones.

3. **Las Evaluaciones del LLM:**
   - `extracted_criteria`: Diccionario `dict[str, list[Criterion]]`. Mapea cada `nct_id` a la lista de criterios parseados. Un `Criterion` tiene `text`, `type` (inclusion/exclusion).
   - `criterion_evals`: Diccionario `dict[str, list[CriterionEval]]`. Aquí se guarda la magia. Cada `CriterionEval` tiene el campo `status` (`"met"`, `"not_met"`, `"NEI"`), el campo `evidence` (la cita textual) y la confianza del LLM.

---

## 2. El Flujo de Nodos: `agent/nodes.py` y `agent/graph.py`

El grafo no se ejecuta linealmente, se orquesta mediante nodos de Python.

### Nodo 1: `parse_patient` y `normalize_mesh`
- **¿Qué hace?** Llama a Azure OpenAI pidiendo un `response_format` que obliga a devolver la estructura `PatientProfile`. Luego, el nodo MeSH busca mapeos en la base de datos local XML.
- **Flujo Crítico:** Si el paciente no menciona edad o sexo, `PatientProfile` lo marca como nulo, y los filtros duros más adelante tienen que ignorarlo por precaución.

### Nodo 2: `retrieve_lexical` (BM25)
- **Implementación:** Reside en `retrieval/bm25.py` y `fielded_bm25.py`.
- **Datos In:** Usa `state["search_plan"]` y `state["mesh_terms"]`.
- **Matemática Fielded:** Aplica pesos a los campos de Elasticsearch/Pyserini. Si un término coincide en el "título" del ensayo, multiplica el score de BM25.
- **Datos Out:** Carga `bm25_candidates` con 1000 IDs y sus `retrieval_norm` (scores de BM25 normalizados entre 0 y 1).

### Nodo 3: `entity_negation_rerank`
- **Por qué existe:** BM25 sufre de ceguera al contexto. Si el ensayo pide "NO diabetes" y el paciente tiene diabetes, BM25 puntúa alto.
- **Código y Algoritmo:** Ubicado en `retrieval/entity_negation.py`. Cruza los diccionarios `diagnoses` del paciente con los campos extraídos del ensayo. Si la librería de NLP (NegSpacy/RegEx) detecta una negación en el ensayo para un término del paciente, aplica un `negation_penalty`.
- **Salida:** Se actualiza `fused_candidates` u ordena de nuevo los candidatos, empujando los penalizados al fondo.

### Nodo 4: Selección de Candidatos (`candidate_selection.py`)
- **El Cuello de Botella:** Aquí se recorta la lista de 1000 a solo 10 (la constante `output_k`).
- **Política `top_score`:** Coge los 10 con mayor score blended. 
- **Política `diverse_top10` (Maximal Marginal Relevance):** Selecciona el mejor, luego busca el siguiente que sea bueno pero ortogonal (diferente) al primero usando embeddings. *Nota: Esto se dejó apagado en P12 porque degradaba el NDCG, aunque subía el recall.*

### Nodo 5: `evaluate_eligibility` (El Cerebro)
- **Implementación:** Usa `eligibility/llm_evaluator.py`.
- **Prompts:** Construye un prompt *few-shot* metiendo ejemplos de cómo un médico razonaría. Le pasa el perfil clínico y la lista de criterios del ensayo.
- **Resultado:** Pydantic genera `TrialEval`. Cada `TrialEval` tiene `overall_status` y una lista de evaluaciones individuales `criterion_evals`. Si un solo criterio de exclusión es `not_met`, el `overall_status` tradicional lo hunde.

### Nodo 6: El Ranking Final (`scorer.py` y P12)
- Ubicado en `ranking/scorer.py`.
- **P11 (Baseline):** Si `trial_eval.overall_status` es "not_eligible", el `base_score` se fija en un número negativo masivo (-1.0). Es un sistema de "Veto duro".
- **P12 (Criterion Evidence Adjust):** Es el avance algorítmico principal. El código itera sobre `criterion_evals`. 
  - Si `status == met`, suma puntos calculando `weight * confidence`. 
  - Genera un `criterion_evidence_score` continuo (ej. `0.85`, `-0.10`).
  - *Suma* este valor al score base (BM25 + Demografía). 
  - **Crítica Positiva:** Esto salva a ensayos excelentes que el LLM castigó por error en un solo criterio menor. Hace que la curva de ranking sea continua (NDCG) en lugar de escalonada.

### Nodos Auxiliares: `generate_questions` y `build_dossiers`
- **Preguntas:** Itera sobre todos los `CriterionEval` donde `status == "NEI"`. Llama al LLM para formular una pregunta. Mapea la pregunta al objeto `ClinicalQuestion` y la guarda en el estado.
- **Dossier:** Usa Jinja2 (`dossier/templates/trial.md.j2`) para inyectar las variables del estado (`nct_id`, `score`, listas de criterios) y renderiza el Markdown final.

---

## 3. Análisis Crítico de Componentes y Vulnerabilidades

Como ingeniero auditando esto, veo los siguientes puntos brillantes y áreas de riesgo:

### Lo Brillante
1. **La Estructura Inmutable de Pydantic:** El hecho de que cada nodo confíe ciegamente en que `PatientProfile` siempre tendrá un campo `.demographics.age` o que será `None` evita cientos de `KeyErrors`.
2. **Rescate del "Tail" (`ranking/fill.py`):** Hay un script de relleno inteligente. Si los 10 candidatos principales fallan estrepitosamente, el sistema va al *tail* (los ensayos de la posición 11 a 1000 del BM25) y los evalúa sintéticamente usando `scorer.score_trial` para rellenar el Top 20 sin tener que gastar tokens del LLM de nuevo.
3. **El Scoring Multinivel:** En `scorer.py`, la matemática es soberbia. Hay premios por la fase del ensayo (`Phase 3` da más puntos que `Phase 1`), estado (`Recruiting` > `Not yet recruiting`), geografía y alineación demográfica.

### Los Puntos Críticos (Vulnerabilidades del Diseño)
1. **Selection Loss Silenciosa:** Cuando `candidate_selection.py` corta de 1000 a 10, si la entidad negada falla o el BM25 ponderó mal, un ensayo "Gold Standard" puede quedarse en la posición 11. El LLM jamás lo leerá, y por tanto, su evaluación real jamás existirá. Este es el límite duro de vuestro Recall@20 (que se clavó en 0.094).
2. **Dependencia Total de la Semántica Pydantic:** Si la API del LLM decide rechazar el esquema JSON (timeout, max tokens o *refusal*), el nodo de evaluación colapsa. Hay que tener mecanismos de reintentos muy fuertes (que LangGraph gestiona parcialmente si hay `retry_edges`, pero a veces fallan).
3. **Falsos Positivos en Filtros Duros:** Si el `ctgov_parser` extrajo que un ensayo era de "Fase 4" basándose en metadata sucia del XML, y el filtro duro está configurado para "Fase 1-3", se descarta un ensayo bueno sin remedio.

---

## 4. Estructura de Salida Final (Qué devuelve el sistema)

El sistema finaliza escribiendo el archivo `predictions.json`. Un registro en este JSON se ve así:

```json
"NCT04211974": {
    "score": 1.565641,
    "rank": 1,
    "eligibility_status": "eligible",
    "evidence_adjust": 0.1358,
    "criterion_breakdown": {
        "met": 3,
        "not_met": 0,
        "nei": 1
    },
    "questions": [
        "¿Cuál es el valor del antígeno carcinoembrionario (CEA)?"
    ]
}
```
**Análisis de esta salida:**
Es un JSON hiper-limpio que permite a un front-end pintar directamente un panel para el médico, ordenado por `score`, indicando rápidamente si es elegible o no, y mostrando las alertas (las preguntas).

---

## 5. Conclusión de la Auditoría

El código de **Trial Matcher** exhibe una madurez técnica equivalente a la de un proyecto en producción de una startup de HealthTech. 
- La arquitectura de **LangGraph** separa correctamente la extracción (NLP) de la matemática de recomendación (Retrieval/Ranking).
- El algoritmo **P12** compensa las carencias estocásticas de los LLMs usando matemáticas de certidumbre continuas.
- El cuello de botella (*Recall Limit*) es una decisión consciente de negocio (Trade-off: Coste de LLM vs Cobertura), no un fallo de código.
