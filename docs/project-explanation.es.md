# Explicación Completa y Exhaustiva del Proyecto: Trial Matcher

Este documento es una guía profunda y detallada creada para explicar el funcionamiento íntegro de **Trial Matcher**, abarcando la arquitectura, la lógica del flujo de datos, los fundamentos técnicos, los algoritmos matemáticos y cada parte del código fuente.

Trial Matcher es un agente inteligente (IA) diseñado para conectar pacientes con los ensayos clínicos más adecuados para ellos, mediante la evaluación semántica de los criterios médicos.

---

## 1. Fundamentos y Arquitectura Principal (El Paradigma de Grafo)

El corazón de este sistema no es un script de ejecución lineal, sino un **Grafo de Estado** (Máquina de Estados Finita) impulsado por **LangGraph**.
El sistema define *Nodos* (funciones o acciones específicas) y *Aristas* (transiciones condicionales entre esos nodos).

### ¿Por qué se usa un Grafo?
Porque emparejar pacientes con ensayos es un proceso complejo. Si el modelo de lenguaje (LLM) falla, o si una búsqueda no devuelve resultados, un script lineal se caería. Con LangGraph, el programa puede retroceder, tomar rutas alternativas o reintentar partes específicas sin perder el contexto.

### El Estado Global: `AgentState`
Toda la información del paciente y el progreso del análisis se guarda en un estado global llamado `AgentState`. Este estado viaja por cada nodo del grafo:
- `patient_raw` / `patient_profile`: El texto base del paciente y el perfil estructurado extraído.
- `search_plan`: El plan de búsqueda generado.
- `bm25_candidates` / `fused_candidates` / `final_candidates`: Listas dinámicas de ensayos potenciales.
- `extracted_criteria` y `criterion_evals`: Criterios médicos extraídos del ensayo y cómo el paciente los cumple o no.
- `ranked_trials`: Lista final ordenada matemáticamente.

---

## 2. El Flujo de Ejecución (Paso a Paso del Código)

Todo el flujo ocurre en `src/trial_matcher/agent/nodes.py`. Cada nodo recibe el estado actual, ejecuta su tarea, y devuelve una pequeña actualización que el grafo une al estado global.

### Etapa 1: Comprensión del Paciente
1. **Nodo `parse_patient`**: Utiliza un modelo de IA (LLM a través de Azure OpenAI) para leer el texto libre que describe al paciente y extraerlo en un objeto rígido y tipado (gracias a **Pydantic**). Se extrae: edad, sexo, diagnósticos, biomarcadores, comorbilidades y tratamientos previos. Esto es vital para no depender de texto ambiguo en los pasos posteriores.
2. **Nodo `normalize_mesh`**: Busca las enfermedades del paciente en la ontología médica oficial (MeSH). Por ejemplo, si el paciente dice "ataque cardíaco", el sistema lo mapea automáticamente al término oficial "Infarto de Miocardio", junto con sus sinónimos.

### Etapa 2: Planeación y Búsqueda (Retrieval)
1. **Nodo `plan_search`**: Toma el perfil y los términos MeSH y crea una estrategia (qué buscar imperativamente y qué incluir como "recomendable").
2. **Nodo `retrieve_lexical` (BM25)**: En vez de enviar cientos de miles de ensayos al LLM (lo cual sería lentísimo y carísimo), se usa el algoritmo clásico y ultrarrápido de búsqueda léxica **BM25** (a través de Elasticsearch/Pyserini). 
   - *Fundamento:* BM25 cuenta la frecuencia de las palabras clave del paciente en los documentos del ensayo. Sin embargo, usa una técnica llamada **"Fielded"** (Campos con pesos): Si la palabra clave aparece en el "Título" o en la "Condición Médica" del ensayo, multiplica matemáticamente su puntuación. Devuelve hasta 1000 candidatos iniciales.

### Etapa 3: Filtrado Inteligente y Reranking
De los 1000 ensayos recuperados, se aplican filtros drásticos:
1. **Filtros Duros (Hard Filters)**: Si un ensayo dice explícitamente "Solo Mujeres" y el paciente es hombre, el ensayo es expulsado inmediatamente de la memoria para ahorrar tiempo.
2. **Nodo `apply_hard_filters` (Entity Negation Rerank)**: (Ubicado en `retrieval/entity_negation.py`). Aquí entra un algoritmo NLP clave que no usa LLM (por velocidad):
   - *El Problema:* BM25 no entiende que un ensayo que dice "NO tener Diabetes" contiene la palabra "Diabetes". Si el paciente tiene Diabetes, BM25 puntuaría alto ese ensayo.
   - *La Solución:* El algoritmo cruza las enfermedades del paciente con las del ensayo. Si la librería NLP (Spacy/RegEx) detecta negaciones en el texto del ensayo ("no asma", "absence of..."), penaliza drásticamente la puntuación (la reduce un 25% o más).
3. **Selección de Candidatos (`candidate_selection.py`)**: Reduce la lista final de 1000 a los mejores **10 ensayos**. Sólo a estos 10 se les dedicará el procesamiento "caro" y profundo del LLM.

### Etapa 4: Razonamiento Médico Profundo (Eligibility)
Este es el cerebro clínico del sistema (Nodo `evaluate_eligibility`, en `eligibility/llm_evaluator.py`).
Por cada uno de los 10 ensayos:
1. El LLM extrae cuidadosamente los Criterios de Inclusión y Exclusión del texto del ensayo.
2. Compara el perfil del paciente con **CADA CRITERIO** y devuelve un veredicto estructurado por Pydantic:
   - `met`: Cumple.
   - `not_met`: No cumple / Excluido (Veto).
   - `NEI` (Not Enough Information): El ensayo pide un dato que no está en la historia clínica del paciente.

### Etapa 5: Puntuación Matemática (Ranking - Algoritmo P12)
Una vez evaluados los 10 ensayos, hay que ordenarlos del 1 al 10. Esto ocurre en `ranking/scorer.py`.
- *Fundamento Matemático:* El sistema calcula un `base_score` (sumando bonus por alineamiento demográfico, fase del ensayo, cercanía geográfica y frescura).
- **Ajuste P12 (Criterion Evidence Adjust)**: Antiguamente, si el LLM ponía "not_met" en un criterio mínimo, el ensayo bajaba su puntuación a -1.0 (se desechaba). El problema es que el LLM puede equivocarse en zonas ambiguas. El algoritmo P12 usa el nivel de **confianza** y la **evidencia** del LLM:
  - **Inclusion Support**: Suma fracciones de puntos proporcionales a los criterios que el paciente sí ha cumplido fehacientemente.
  - **Exclusion Penalty**: Resta puntos si hay un fallo, pero dependiendo de la confianza.
  El resultado es una puntuación matemática continua (ej. 1.56, 0.85, -0.10) mucho más realista.

### Etapa 6: Cierre Clínico y Entrega (Output)
1. **Nodo `generate_questions`**: Por cada criterio evaluado como `NEI` (falta de información) en los ensayos principales, el LLM genera una pregunta directa y concisa para el médico (ej. *"¿Cuál es la medida exacta de hemoglobina del paciente?"*).
2. **Nodo `build_dossiers`**: Utilizando plantillas Jinja2, se empaqueta toda la evidencia (puntuaciones, tablas de criterios cumplidos y no cumplidos, justificaciones de la IA) en hermosos informes Markdown listos para ser leídos por el especialista humano.

---

## 3. Comprendiendo las Métricas de Evaluación

El sistema se mide contra un Benchmark (TREC 2021) usando las siguientes métricas, y es importante entender qué hace cada una:
- **Recall@20 (Potencia del Buscador)**: Mide cuántos de los ensayos verdaderamente relevantes a nivel mundial lograste colar en tu Top 20. Si el motor inicial BM25 falló al buscarlos, el Recall será bajo.
- **Micro-F1 (Inteligencia del LLM)**: Mide el porcentaje de veces que el LLM acertó al decir que el paciente cumplía (`met`) o no (`not_met`) las reglas del ensayo, frente a la respuesta oficial de los médicos.
- **NDCG@10 (Calidad del Ordenamiento)**: Es la métrica reina. No solo evalúa si encontraste buenos ensayos, sino si los colocaste exactamente en la posición 1, 2 y 3 de tu ranking (premiando la posición alta).

---

## 4. Estructura de Directorios Clave

- `src/trial_matcher/agent/`: Donde viven los **nodos del grafo** y la arquitectura de LangGraph.
- `src/trial_matcher/retrieval/`: Donde reside la lógica matemática de búsqueda rápida (BM25) y la **penalización de negaciones** (`entity_negation.py`).
- `src/trial_matcher/eligibility/`: Código que envía Prompts (textos guía) complejos al LLM usando Pydantic, forzando respuestas JSON matemáticas y deterministas.
- `src/trial_matcher/ranking/`: Los ficheros matemáticos (`scorer.py`) que aplican pesos a los criterios de inclusión/exclusión para generar el orden final.
- `src/trial_matcher/models/`: Clases (Pydantic y TypedDicts) que garantizan que el sistema jamás rompa por un "error de tipado". Aquí vive `AgentState`.

## Resumen Final

Trial Matcher fusiona inteligentemente dos mundos:
1. **Fuerza bruta matemática (Escalabilidad)**: BM25 y Entity Negation criban cientos de miles de ensayos en 1 segundo y a coste $0.
2. **Razonamiento Semántico (Precisión Clínica)**: Los LLM analizan meticulosamente el top 10 resultante como lo haría un humano, apoyándose en Pydantic y Grafos para no descarrilar.
3. **Sistemas de Seguridad (Scoring)**: Las fallas del LLM son amortiguadas por algoritmos matemáticos como el "P12 Criterion Evidence", que asegura que no se pierdan buenas oportunidades por alucinaciones leves de la IA.
