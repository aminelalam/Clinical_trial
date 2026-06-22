# Batería de Preguntas y Respuestas para la Defensa del Proyecto (Trial Matcher)

Este documento contiene un simulacro exhaustivo de preguntas que un tribunal técnico, médico o académico podría hacerte durante la defensa de Trial Matcher. 
Están divididas desde un nivel superficial (producto) hasta un nivel técnico extremo (ingeniería, grafos, matemáticas y LLMs). Estudiar este documento te garantiza tener respuesta para cualquier cosa.

---

## BLOQUE 1: Visión de Producto y Justificación (Superficial - Nivel Negocio/Médico)

### 1. ¿Cuál es el objetivo principal de este proyecto en una frase?
**Respuesta:** Resolver el problema del reclutamiento ineficiente en ensayos clínicos creando un agente autónomo que cruza perfiles de pacientes con una base de datos global de estudios, evaluando automáticamente si el paciente es elegible o no, y justificando médicamente la decisión.

### 2. ¿Por qué usar IA Generativa y no un buscador clásico con filtros (como buscar piso en Idealista)?
**Respuesta:** Porque el lenguaje clínico es sumamente complejo y está lleno de negaciones e inferencias temporales. Un buscador clásico (BM25) busca palabras exactas; si un ensayo dice *"Excluir pacientes con historial de asma"*, y el paciente pone en su historial *"Sufro de asma"*, un buscador clásico los empareja porque ambos textos contienen la palabra "asma". La IA generativa actúa como un médico lector: comprende que una es una condición de exclusión y la otra es una condición activa, descartando el ensayo correctamente.

### 3. ¿De dónde salen los datos? ¿Qué es TREC Clinical Trials?
**Respuesta:** TREC (Text Retrieval Conference) es la cumbre mundial de evaluación de motores de búsqueda organizada por el gobierno de EE.UU. Hemos usado su edición de 2021 porque nos provee dos cosas invaluables: el *snapshot* real de la base de datos de ClinicalTrials.gov de aquel momento (con 370.000 ensayos) y un archivo oficial de "Qrels" (Juicios de Relevancia), donde médicos humanos determinaron qué ensayos eran realmente buenos para 75 pacientes de prueba. Usamos esto último para medir si nuestro agente funciona sin hacernos trampas al solitario.

---

## BLOQUE 2: Arquitectura e Ingeniería de Software (Nivel Medio - Arquitectura)

### 4. ¿Por qué usar la librería LangGraph en lugar de un simple script de Python en cadena?
**Respuesta:** En la ingeniería de LLMs, las cadenas de ejecución simples (LangChain puro) son frágiles y opacas. Usamos **LangGraph** porque modela el proceso como una **Máquina de Estados Finita** (Nodos y Aristas).
- **Aislamiento:** Si la extracción de MeSH falla, el fallo se contiene en un nodo.
- **Flujos cíclicos:** Si un nodo de evaluación detecta que no hay candidatos, el grafo podría "retroceder" dinámicamente al nodo de búsqueda para relajar los parámetros.
- **Trazabilidad:** Nos permite inyectar observabilidad, viendo exactamente qué variables entran y salen en cada paso de la mente del agente.

### 5. Hablando de grafos... ¿Qué hay exactamente en la memoria del agente? ¿Qué es el `AgentState`?
**Respuesta:** El `AgentState` es una estructura de datos estricta (una clase en Python) que viaja de nodo en nodo. Sus campos exactos son:
1. `patient`: Un objeto Pydantic con el texto original y la información demográfica (edad, sexo).
2. `mesh_terms`: Lista de conceptos médicos normalizados que extraemos al principio.
3. `search_plan`: Objeto que dice qué queries exactas le vamos a lanzar al buscador.
4. `retrieved_trials`: Todos los ensayos en bruto que devolvió el buscador léxico (ej. 1000 IDs).
5. `final_candidates`: La lista hiper-reducida (10 ensayos) que sobrevivieron a los filtros rápidos.
6. `ranked_trials`: La estructura final, que ya incluye las respuestas estructuradas del LLM (`met`, `not_met`, `NEI`), la puntuación matemática final y las listas de evidencias médicas.

### 6. ¿Cómo forzáis al LLM a que no "alucine" o conteste de forma impredecible?
**Respuesta:** No usamos al LLM como un chatbot que devuelve un párrafo de texto. Utilizamos **Pydantic** y el formato de **Structured Outputs** de las APIs modernas. Cuando le pedimos al LLM que evalúe la elegibilidad de un ensayo, le forzamos, a nivel de la API, a devolver un JSON con un esquema estricto (por ejemplo, una lista de objetos `CriterionEvaluation` que deben contener obligatoriamente un campo `status` que SOLO puede tener tres valores: `"met"`, `"not_met"`, o `"NEI"`). Si el LLM intenta devolver otra cosa, la librería lanza un error de parseo.

---

## BLOQUE 3: Flujo de Ejecución y Matemáticas (Nivel Profundo - Data Science)

### 7. Decís que usáis BM25 Fielded al principio. ¿Por qué "Fielded" y cómo es su matemática?
**Respuesta:** BM25 es una fórmula estadística (basada en Term Frequency e Inverse Document Frequency) que puntúa documentos. Nosotros usamos **Fielded BM25** porque un ensayo clínico tiene estructura. No vale lo mismo que la palabra "Cáncer de Mama" aparezca en el resumen largo, a que aparezca explícitamente en el campo *Condición*. Al indexar, aplicamos **pesos multiplicadores** a ciertos campos (por ejemplo, el campo de condición o los títulos reciben un peso mayor en la ecuación) garantizando que los resultados del motor de búsqueda sean semánticamente más precisos antes de llegar al LLM.

### 8. Explicadme el cuello de botella. BM25 devuelve 1000 ensayos, pero al LLM solo le mandáis 10. ¿No es un riesgo enorme?
**Respuesta:** Es el riesgo más grande del sistema, conocido como *Selection Loss* (Pérdida de selección). Si mandásemos los 1000 al LLM, tendríamos un sistema perfecto, pero tardaría horas por paciente y costaría una fortuna en tokens de API.
Para paliar el riesgo de que los 10 mejores ensayos se nos queden fuera, usamos heurísticas intermedias antes del LLM:
1. **Hard Filters:** Cortamos rápidamente los ensayos que matemáticamente sabemos que no encajan por el sexo o la edad extraída del paciente.
2. **Entity Negation Rerank:** Un algoritmo rápido en Python (sin LLM) que recalcula un *blended_score* usando la presencia de términos superpuestos.

### 9. Vuestro algoritmo P12 (*Criterion Evidence Adjust*). ¿Por qué cambiasteis la versión 11 por la 12 en la fase de ranking?
**Respuesta:**
- En la versión **P11 (Baseline Clásico)**, teníamos una regla rígida: si el LLM ponía tan solo un criterio de todo el ensayo como `not_met` (no cumple), el ensayo quedaba vetado matemáticamente y lo hundíamos en el ranking penalizándolo (Score -1.0). El problema es que los LLMs cometen errores, y un solo error destruía ensayos que en la realidad médica eran el estándar de oro.
- En la versión **P12**, diseñamos un modelo basado en Evidencia (inspirado en TrialGPT). En lugar de vetos absolutos, aplicamos **ajustes progresivos**. Si un ensayo tiene mucho *Inclusion Support* (ej. el paciente cumple 8 criterios clave) sumamos puntos base. Si hay una exclusión, penalizamos en función del nivel de certidumbre. Esto hizo que el sistema fuera **más resistente a las alucinaciones del LLM**, disparando el NDCG@10 hasta 0.520.

---

## BLOQUE 4: Métricas y Resultados (Nivel Tribunal / Defensa Final)

### 10. ¿Por qué usáis NDCG@10 como métrica reina y no Accuracy o Precision?
**Respuesta:** Porque estamos evaluando un sistema de **Ranking**, no un clasificador binario.
En un buscador médico, que el ensayo correcto aparezca en la posición 1 o en la posición 9 cambia la vida del paciente (el médico no lee más allá de las 3 primeras opciones). **NDCG (Normalized Discounted Cumulative Gain)** premia que los ensayos "más relevantes" (clasificados como nivel 2 por los jueces) estén en las posiciones más altas de la lista, dividiendo los puntos por el logaritmo de la posición (descuento). Una simple Precisión solo nos diría cuántos aciertos hubo en el Top 10, pero no si estaban ordenados correctamente.

### 11. ¿Y qué es la métrica Micro-F1?
**Respuesta:** El Micro-F1 evalúa el "cerebro" del LLM midiendo su desempeño en la tarea de extracción de Elegibilidad (La Tarea 2 del reto). Mide cuán acertadamente predice las etiquetas (`met`, `not_met`, `NEI`) teniendo en cuenta tanto falsos positivos como falsos negativos de manera global a través de todas las clases. Obtener un ~0.50 significa un nivel altísimo de comprensión lectora clínica automatizada dada la inmensa ambigüedad del benchmark TREC.

### 12. Generación de Preguntas y Dossier (T4 y T5). ¿Cómo medís la calidad si no hay humanos puntuándolo?
**Respuesta:** Mediante **Métricas Heurísticas Automatizadas**.
- Para las **Preguntas (T4)**, hemos escrito un *script de evaluación* que comprueba 5 ejes matemáticamente/mediante reglas semánticas: ¿La pregunta contiene el término clínico que originó el "NEI"? ¿Acaba en signo de interrogación? ¿Pide un marco temporal? Si cumple todo, recibe 1.0. Nuestra media fue 0.91.
- Para los **Dossiers (T5)**, el *script de evaluación* escanea el Markdown generado buscando los 8 elementos obligatorios de la rúbrica (NCT_ID, tabla de elegibilidad, resumen, flags, enlace web). Si un dossier está completo estructuralmente, tiene 1.0. Obtuvimos un 0.90.

---

## BLOQUE 5: Trabajo Futuro y Limitaciones (Sinceridad Técnica)

### 13. ¿Qué le falta al proyecto para poder ser usado en un hospital real mañana?
**Respuesta:** Tiene tres grandes limitaciones que tenemos identificadas:
1. **Falta de Retrieval Denso (Vectores):** Actualmente usamos BM25 (Búsqueda por palabra clave). Aunque BM25 es rápido, no captura la verdadera intención semántica. El siguiente paso lógico es integrar "Dense Retrieval" (usar Embeddings de modelos pre-entrenados como BGE o ClinicalBERT) para buscar por conceptos matemáticos en el espacio latente.
2. **Intervención Humana "Human-in-the-loop":** El agente final hoy es no interactivo. En un entorno hospitalario, la generación de la pregunta (NEI) debería saltar en la pantalla del médico, para que él responda con el dato faltante, y el grafo vuelva a recalcular el ranking en vivo.
3. **Escala de Procesamiento de Candidatos:** Subir de 10 a 50 ensayos evaluados por paciente dispararía nuestro Recall final, pero requeriría desplegar modelos en hardware dedicado por razones económicas.

### 14. Me he fijado que en el repositorio hay referencias a 'HyDE', 'Dense Retrieval' y 'Self-Consistency'. ¿Por qué todo esto está apagado en la entrega final?
**Respuesta:** Porque en investigación es crucial establecer un **Baseline Sólido, Rápido y Transparente**. Todos esos módulos (HyDE para alucinación controlada, Evaluadores Secundarios para revisar la salida del primero) consumen 5 veces más tokens y añaden muchísima latencia.
Decidimos dejarlos "apagados" porque queríamos demostrar que, logrando nuestro NDCG de 0.52 con un sistema puro de búsqueda básica y una sola pasada del LLM, poseemos una base algorítmica impecable. Demuestra que nuestro código funciona porque la arquitectura es buena, no porque tiramos "fuerza bruta computacional y dinero" al problema. Esos módulos inactivos son la demostración de todo el techo de mejora inmediata que tiene el proyecto de cara al futuro.
