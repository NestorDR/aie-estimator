# Informe de Pruebas de Estrés — Línea Base CAG

- **Modelo / Perfil**: `reasoner-fast (openai/gpt-6-luna priorizado)` (`--http http://127.0.0.1:8000`, `--repeats 1`)
- **Dataset**: [`evals/stress/results.csv`](./results.csv) — **65 turnos** ejecutados (**0 errores**)
- **Coste total**: **\$0.1149** (`65/65` turnos cumplieron el presupuesto de \$0.05/turno)
- **Presupuesto de latencia (8,000 ms)**: **4/65** turnos cumplieron (`61/65` superaron los 8 segundos)

## Tabla Resumen

| Escenario / Prueba | KB | Turnos | P50 ms | P95 ms | Coste Total ($) | Cache Hit % | Recuperación Media (Recall) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attachment_0kb | 0 | 1 | 6261 | 6261 | 0.0003 | 0.0% | n/a |
| attachment_100kb | 100 | 1 | 8646 | 8646 | 0.0006 | 0.0% | 0.0% |
| attachment_20kb | 20 | 1 | 7759 | 7759 | 0.0006 | 0.0% | 0.0% |
| attachment_50kb | 50 | 1 | 6353 | 6353 | 0.0005 | 0.0% | 0.0% |
| attachment_5kb | 5 | 1 | 8797 | 8797 | 0.0007 | 0.0% | 0.0% |
| contradiction | 0 | 20 | 11875 | 17194 | 0.0309 | 0.0% | 0.0% |
| growing | 0 | 20 | 15797 | 17991 | 0.0418 | 0.0% | 100.0% |
| pivot | 0 | 20 | 15637 | 19457 | 0.0396 | 0.0% | 0.0% |

## Curva 1: Latencia vs Tokens de Entrada

| Rango (`tokens_in`) | Turnos | Media `tokens_in` | P50 ms | P95 ms |
| :--- | ---: | ---: | ---: | ---: |
| 1000-2999 | 5 | 2638 | 8797 | 10300 |
| 3000-9999 | 32 | 6925 | 12749 | 16426 |
| >=10000 | 28 | 11760 | 17442 | 19863 |

## Curva 2: Coste Acumulado vs Turno (Multi-Turno)

| Turno | growing ($) | pivot ($) | contradiction ($) |
| ---: | ---: | ---: | ---: |
| 1 | 0.0011 | 0.0006 | 0.0005 |
| 2 | 0.0023 | 0.0016 | 0.0010 |
| 3 | 0.0037 | 0.0030 | 0.0017 |
| 4 | 0.0051 | 0.0045 | 0.0025 |
| 5 | 0.0067 | 0.0061 | 0.0033 |
| 6 | 0.0084 | 0.0078 | 0.0043 |
| 7 | 0.0104 | 0.0095 | 0.0055 |
| 8 | 0.0127 | 0.0116 | 0.0072 |
| 9 | 0.0150 | 0.0138 | 0.0088 |
| 10 | 0.0174 | 0.0159 | 0.0105 |
| 11 | 0.0198 | 0.0182 | 0.0123 |
| 12 | 0.0221 | 0.0206 | 0.0142 |
| 13 | 0.0245 | 0.0230 | 0.0160 |
| 14 | 0.0269 | 0.0255 | 0.0180 |
| 15 | 0.0290 | 0.0281 | 0.0200 |
| 16 | 0.0315 | 0.0297 | 0.0220 |
| 17 | 0.0340 | 0.0324 | 0.0241 |
| 18 | 0.0365 | 0.0351 | 0.0263 |
| 19 | 0.0391 | 0.0375 | 0.0286 |
| 20 | 0.0418 | 0.0396 | 0.0309 |

## Curva 3: Tasa de Recuperación (Recall) de `MemoryDriftMetric` vs N

| Turno de control (N) | growing | pivot | contradiction |
| ---: | ---: | ---: | ---: |
| 1 | n/a | n/a | n/a |
| 3 | 100.0% | 0.0% | 0.0% |
| 6 | 100.0% | 0.0% | 0.0% |
| 10 | 100.0% | 0.0% | 0.0% |
| 20 | 100.0% | 0.0% | 0.0% |

## Dónde Empieza a Romper Nuestro CAG y Por Qué

**1. El tiempo de respuesta rompe antes que el presupuesto:** Aunque cada turno fue barato (`65/65` cumplieron el límite de \$0.05/turno, con un gasto total de **\$0.1149** en 65 turnos), **61 de los 65 turnos superaron el objetivo de 8,000 ms**. En cuanto el historial supera los 3,000 tokens de entrada, la latencia mediana pasa de **8.8 s** (`1000-2999` tokens) a **12.7 s** (`3000-9999` tokens) y **17.4 s** (`>=10000` tokens, con P95 de **19.9 s**). Para cualquier usuario en un chat real, esperar 15–20 segundos por mensaje resulta lento; además, como las sesiones conversacionales omiten ambas cachés (`0.0%` de aciertos) y lanzan llamadas extra de resumen y extracción de metadatos a partir del turno 7, la latencia sigue subiendo incluso usando el perfil de LLMs más rápido (`reasoner-fast`).

**2. Los archivos grandes se recortan y los datos iniciales son frágiles:** Al adjuntar PDFs, el sistema limita el texto extraído a **12,000 caracteres**, descartando **4,378 caracteres** en `20 KB`, **28,930 caracteres** en `50 KB` y **69,853 caracteres** en `100 KB`, lo que provocó un **0.0%** de tasa de recuperación (*recall*) en todos los tamaños (`5–100 KB`). En conversaciones largas, cada escenario introduce en el turno 1 el nombre del proyecto y se comprueba si sigue guardado en `metadata.project_name`: `growing` mantuvo `"Nimbus"` con **100.0%** de tasa de recuperación en los 20 turnos, mientras que `pivot` (`"Helios"`) y `contradiction` (`"Atlas"`) cayeron a **0.0%** desde el turno 2 porque el extractor de metadatos no capturó esos nombres tras el turno. Además, las anclas de memoria solo se activan a partir del turno 7 para compromisos contractuales o de cumplimiento normativo, por lo que las presentaciones iniciales del proyecto nunca se anclaron (`anchors_count = 0` en los 65 turnos). En términos sencillos: el enfoque CAG actual funciona bien en conversaciones cortas y estables, pero pierde información cuando los documentos superan ~12k caracteres o cuando el extractor de metadatos omite un dato inicial.
