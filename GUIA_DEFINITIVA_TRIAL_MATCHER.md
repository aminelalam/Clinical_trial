# Guía Definitiva y Exhaustiva de Trial Matcher

Este documento es la **radiografía técnica absoluta** del proyecto Trial Matcher. Su objetivo es que cualquier persona (evaluador, ingeniero o investigador) pueda leerlo de principio a fin y comprender al 100% no solo *qué* hace el código, sino *por qué* lo hace, *cómo* lo hace, y la matemática y lógica subyacente de cada algoritmo.

---

## Índice
1. [Visión General y Filosofía del Proyecto](#1-visión-general-y-filosofía-del-proyecto)
2. [Arquitectura Base: El Paradigma de Grafo (LangGraph)](#2-arquitectura-base-el-paradigma-de-grafo-langgraph)
3. [El Flujo de Vida de un Paciente (Paso a Paso)](#3-el-flujo-de-vida-de-un-paciente-paso-a-paso)
4. [Algoritmos Core y Modelos Matemáticos](#4-algoritmos-core-y-modelos-matemáticos)
5. [Métricas de Evaluación: Entendiendo el Rendimiento](#5-métricas-de-evaluación-entendiendo-el-rendimiento)
6. [Estructura del Repositorio](#6-estructura-del-repositorio)

---

## 1. Visión General y Filosofía del Proyecto

### El Problema
El reclutamiento ineficiente arruina ensayos clínicos reales. Los pacientes no saben que existen ensayos para ellos, y los médicos no tienen tiempo de leer manuales de 50 páginas de criterios de inclusión/exclusión. 
Hacer un simple buscador por palabras (Google) no sirve: si un ensayo dice "Excluir pacientes con asma", y el paciente busca "asma", el sistema clásico le recomendará ese ensayo (falso positivo).

### La Solución: Agente Inteligente (Trial Matcher)
Trial Matcher no es un buscador clásico (RAG tradicional). Es un **agente lógico** que:
1. **Entiende** al paciente usando terminología médica.
2. **Busca** en una base de datos gigante (370.000 ensayos).
3. **Razona** como un médico: lee los criterios del ensayo y comprueba si el paciente los cumple, los incumple, o si falta información.
4. **Ordena** las opciones priorizando la evidencia médica.
5. **Redacta** informes (dossiers) listos para que el médico tome la decisión final.

---

## 2. Arquitectura Base: El Paradigma de Grafo (LangGraph)

El corazón de este proyecto no es un *script* lineal, sino una **Máquina de Estados Finita** gestionada por **LangGraph**.

### ¿Por qué LangGraph?
En un pipeline complejo, si el LLM falla, o si no se encuentran ensayos, un script lineal se rompería. LangGraph permite definir **Nodos** (acciones) y **Aristas** (transiciones condicionales). Esto permite:
- **Memoria persistente (State):** Hay un objeto llamado `AgentState` que viaja de nodo en nodo. Todo lo que descubre un nodo (ej. la búsqueda BM25), se guarda en el estado para el siguiente nodo.
- **Control de Flujo:** Podemos tener bucles condicionales (ej. "Si no encontraste ensayos, vuelve al nodo de búsqueda y amplía la query").

### El Estado del Agente (`AgentState`)
Cada vez que procesamos un paciente, se inicializa un "estado" que contiene:
* `patient`: Perfil en texto original y datos estructurados.
* `mesh_terms`: Términos médicos normalizados.
* `search_plan`: La estrategia de cómo vamos a buscar.
* `retrieved_trials`: Lista en bruto de ensayos recuperados.
* `final_candidates`: Lista filtrada de candidatos prometedores.
* `ranked_trials`: Lista final matemática y ordenada de los mejores ensayos.

---

## 3. El Flujo de Vida de un Paciente (Paso a Paso)

Cuando el agente recibe a un paciente (ej. Topic 75), la información viaja por este exacto circuito de nodos:

### Etapa 1: Comprensión (Nodos `parse_patient` y `normalize_mesh`)
1. **Parseo Estructurado:** El texto libre del paciente se envía al LLM para extraer su edad, sexo, diagnósticos, biomarcadores y tratamientos previos usando **Pydantic** (esto fuerza al LLM a devolver un JSON estricto, no texto libre).
2. **Normalización MeSH (Medical Subject Headings):** Las enfermedades tienen sinónimos ("ataque al corazón" = "infarto de miocardio"). El agente busca los términos del paciente en la ontología oficial MeSH para usar un lenguaje médico estándar.

### Etapa 2: Recuperación / Retrieval (Nodos `plan_search` y `retrieve_lexical`)
1. **Planificador:** El agente mira los términos MeSH y decide qué consultar.
2. **Retrieve (BM25):** Para evitar gastar dinero procesando 370.000 ensayos con LLM, usamos búsqueda léxica ultrarrápida (Elasticsearch/BM25). El sistema pide los **1000 ensayos** que más palabras comparten con el paciente. Se recuperan en menos de 1 segundo.

### Etapa 3: Filtrado y Selección Rápida (Nodos `apply_hard_filters` y `entity_negation_rerank`)
Aquí empieza la "criba":
1. **Filtros Duros:** Si el paciente es un hombre de 45 años, cualquier ensayo que requiera ser mujer, o niños menores de 18, es fulminado al instante de la memoria. (Ahorro brutal de cómputo).
2. **Entity Negation Rerank:** (Algoritmo clave). A los ensayos que sobreviven, se les aplica un algoritmo rápido en Python (sin LLM) que comprueba si hay solapamiento de entidades médicas positivas y penaliza si hay negaciones lógicas cruzadas. 
3. **Selección de Candidatos:** De esos 1000 ensayos, el sistema **solo se queda con 10** (Configurado como `max_trials_per_topic=10`). ¿Por qué 10? Porque evaluar ensayos clínicos completos con LLMs (GPT-4/Claude) es carísimo (tokens). Enviamos al LLM solo los 10 con mayor probabilidad matemática de encajar.

### Etapa 4: Razonamiento Profundo (Nodos `extract_criteria` y `evaluate_eligibility`)
Aquí ocurre la magia de la Inteligencia Artificial.
Por cada uno de los 10 ensayos elegidos:
1. **Extracción:** El LLM extrae en bloques la "montaña de texto" de criterios de inclusión y exclusión del ensayo.
2. **Evaluación (Eligibility):** El LLM compara el perfil del paciente contra CADA criterio y emite un veredicto estructurado:
   - `met`: El paciente cumple la regla.
   - `not_met`: El paciente rompe la regla (¡Veto!).
   - `NEI` (Not Enough Information): El ensayo pide algo que no sabemos del paciente (ej. presión arterial).

### Etapa 5: Puntuación Final (Nodo `rank_node` y P12)
Una vez el LLM ha evaluado los 10 ensayos, hay que ordenarlos matemáticamente para entregar el "Top 1" al médico. 
El algoritmo (conocido como **P12 - Criterion Evidence Adjust**) otorga puntos base y aplica multiplicadores según la evaluación (más detalles en la sección de Algoritmos). Al final, la lista de ensayos se ordena de mayor a menor *score*.

### Etapa 6: Cierre Clínico (Nodos `generate_questions` y `build_dossiers`)
1. **Preguntas (T4):** Si en el ensayo ganador salió que había datos `NEI` (Not Enough Info), el LLM genera una pregunta redactada para el médico (ej. *"¿Cuál es la fracción de eyección ventricular del paciente?"*).
2. **Dossiers (T5):** Se genera un archivo `.md` precioso, resumido, con una tabla de los criterios cumplidos, los vetados, las advertencias (*flags*) y la justificación.

---

## 4. Algoritmos Core y Modelos Matemáticos

### Algoritmo de Búsqueda: BM25 Fielded
**¿Qué es?** BM25 (Best Matching 25) es el algoritmo estándar de oro en recuperación de información tradicional (es lo que usa Lucene/Elasticsearch). 
**¿Cómo funciona?** Cuenta la frecuencia de una palabra del paciente (ej. "Glioma") en el documento del ensayo. Cuantas más veces aparece, mayor *score*. PERO, penaliza la longitud del documento (los documentos largos no deben tener ventaja injusta) y penaliza palabras muy comunes ("el", "un").
**¿Por qué "Fielded"?** Porque no buscamos en todo el texto del ensayo por igual. Multiplicamos x5 el peso si la palabra aparece en el campo "Condición Médica" del ensayo, y x1 en el resumen.

### Entity Negation Reranker (Algoritmo de Priorización Rápida)
**Problema:** BM25 no entiende el contexto. BM25 puntúa alto un ensayo que dice "NO tener Glioma" porque la palabra "Glioma" está presente.
**Solución Matemática:** 
Calculamos un `blended_score` mezclando la puntuación de BM25 normalizada y un cálculo de intersección de palabras:
1. `disease_support`: Fracción de términos de la enfermedad del paciente que existen en el ensayo.
2. `negation_penalty`: Si usamos procesamiento de lenguaje natural (Spacy/NegSpacy) y vemos que el ensayo dice "no asma", y el paciente "tiene asma", aplicamos una penalización matemática (ej. multiplicamos el score del ensayo por 0.25, hundiéndolo en la lista).

### Algoritmo de Ranking Final (Baseline vs Ajuste P12)
**El Baseline Clásico:** Inicialmente, el sistema sumaba puntos si un ensayo estaba en Fase 3, reclutando, etc., pero castigaba brutalmente (con penalizaciones sintéticas como un score de `-1.0`) si cualquier criterio era evaluado como `not_met` por el LLM.
**El Problema:** El LLM a veces se equivoca. Evalúa como `not_met` algo que igual era ambiguo, hundiendo ensayos excelentes (pérdida de "oro").
**Ajuste P12 (Criterion Evidence Score):** 
Inspirado en papers recientes (como TrialGPT), el algoritmo final usa los niveles de **confianza** y **evidencia** del LLM.
Fórmula Simplificada:
```text
Score Final = Score Base + (Inclusion Support) - (Exclusion Penalty) - (Missing Data Penalty)
```
- **Inclusion Support:** Se suman puntos (+0.5, +1.0) proporcionales a cuántos criterios de inclusión fuertes ha cumplido el paciente.
- **Exclusion Penalty:** Se restan puntos solo si la exclusión es clarísima (alta confianza de que el paciente está excluido).
Esto permite un ranking mucho más orgánico y elástico, premiando a los ensayos donde hay muchísima evidencia de encaje, incluso si hay una leve duda en una regla menor.

---

## 5. Métricas de Evaluación: Entendiendo el Rendimiento

TREC Clinical Trials evalúa el proyecto en base a una métrica combinada, donde cada una mide algo diferente del pipeline.

### Recall@20 (Cobertura)
**¿Qué significa?** De todos los ensayos "relevantes" que existen en la base de datos (según los jueces humanos de TREC), ¿cuántos logró capturar tu sistema en su Top 20 final?
**Interpretación:** Si hay 10 ensayos buenos en el mundo para el paciente, y tu sistema devuelve 1 entre los 20 primeros, tu recall es bajo. Mide la **potencia del motor de búsqueda (BM25)** y que los filtros duros no estén destruyendo cosas buenas.

### Micro-F1 (Exactitud de Extracción)
**¿Qué significa?** Es la media armónica de la Precisión y el Recall enfocada en las clases de elegibilidad (`met`, `not_met`, `NEI`). 
**Interpretación:** Mide **lo listo que es el LLM**. Si el ensayo dice "solo mujeres" y el paciente es hombre, ¿acertó el LLM poniendo `not_met`? Un Micro-F1 de 0.50 implica que el sistema acierta la lógica médica la mitad de las veces (lo cual es muy alto para la ambigüedad médica).

### NDCG@10 (Normalized Discounted Cumulative Gain - Calidad del Ranking)
**¿Qué significa?** Es la **métrica reina**. Mide si ordenaste bien la lista. No basta con encontrar los ensayos buenos, **tienen que estar arriba del todo**. "Discounted" significa que si pones un ensayo excelente en la posición 10, te da muy pocos puntos, pero si lo pones en la posición 1, te da muchísimos.
**Interpretación:** El NDCG@10 final del proyecto es **0.520**, lo que significa que el algoritmo de Ranking (P12) está haciendo un trabajo fenomenal subiendo a la cima el "oro" que recuperó BM25.

### Métricas Heurísticas (T4 Preguntas y T5 Dossier)
Como TREC no evalúa preguntas y dossiers, se usan algoritmos internos matemáticos:
- **Calidad de Pregunta (T4):** Suma puntos (0 a 1) si la pregunta generada incluye el término médico, tiene formato de pregunta (?), y especifica tiempo (ej. "En los últimos 6 meses"). Resultado: **0.91** (sobresaliente).
- **Completitud Dossier (T5):** Mide si el markdown generado tiene tabla de criterios, resumen ejecutivo y enlaces clínicos. Resultado: **0.90** (sobresaliente).

---

## 6. Estructura del Repositorio

Si abres la carpeta del código, debes entender cómo navegar:

- `src/trial_matcher/agent/`: Aquí viven los Nodos del grafo de LangGraph. El corazón del flujo de control.
- `src/trial_matcher/retrieval/`: Algoritmos de búsqueda. Aquí está la implementación matemática de BM25 y Entity Reranker.
- `src/trial_matcher/eligibility/`: Aquí está el código que invoca al LLM, le pasa la plantilla de evaluación médica y usa Pydantic para forzar respuestas estrictas.
- `src/trial_matcher/ranking/`: Los algoritmos matemáticos (scorer.py, P12) que cogen las evaluaciones y ordenan la lista final.
- `eval/`: Scripts independientes para comparar tus predicciones JSON contra los *qrels* (los juicios humanos oficiales de la universidad/NIST) y calcular el NDCG, Recall, etc.
- `entrega_final/`: La copia inmutable de los artefactos, los informes (`.md`) y el `predictions.json` definitivo.

---
**Conclusión:** Trial Matcher es una pieza de ingeniería agéntica de vanguardia. Separa perfectamente la fuerza bruta de búsqueda escalar (Lexical BM25) del razonamiento semántico profundo (LLMs), fusionándolos con heurísticas médicas deterministas y ranking probabilístico.
