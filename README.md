# Detección multimodal de emergencias en redes sociales

Proyecto de aprendizaje automático multimodal para clasificar publicaciones de Twitter/X como **informativas** o **no informativas** durante desastres naturales.

## Descripción

El proyecto utiliza el dataset público [CrisisMMD](https://huggingface.co/datasets/QCRI/CrisisMMD) y combina dos fuentes de información:

- **Texto:** preprocesamiento de tweets y representación TF-IDF.
- **Imagen:** embeddings visuales obtenidos con CLIP ViT-B/32.
- **Fusión:** concatenación de 768 características textuales y 512 visuales.
- **Clasificador:** red neuronal MLP de 1.280 → 512 → 1.

La finalidad es estudiar si la combinación de texto e imagen mejora la detección de publicaciones útiles para situaciones de emergencia.

## Dataset

Se utilizó la configuración `informative` de CrisisMMD. Para facilitar la ejecución en Google Colab se empleó una muestra estratificada del 10 %:

- 1.806 publicaciones.
- 1.111 informativas.
- 695 no informativas.
- Textos e imágenes asociados.
- Particiones oficiales `train`, `dev` y `test`.

La muestra mantiene la proporción de clases dentro de cada partición oficial.

## Metodología

1. Carga de anotaciones desde Hugging Face.
2. Selección de una muestra estratificada.
3. EDA de clases, eventos, textos e imágenes.
4. Limpieza de tweets: URLs, menciones, retweets, símbolos y stopwords.
5. Extracción textual mediante TF-IDF.
6. Extracción visual mediante CLIP ViT-B/32 congelado.
7. Concatenación de características.
8. Entrenamiento de una red MLP en PyTorch.
9. Detección de sobreajuste mediante curvas de aprendizaje y early stopping.
10. Optimización de hiperparámetros mediante Optuna y repeated stratified k-fold CV.
11. Comparación contra regresión logística.
12. Prueba estadística de Wilcoxon.
13. Visualización del espacio multimodal mediante t-SNE.
14. Evaluación final en el conjunto de prueba.

## Resultados principales

| Modelo | ROC-AUC CV promedio | Desviación estándar |
|---|---:|---:|
| MLP base | 0.8093 | 0.0209 |
| MLP optimizado | 0.8435 | 0.0257 |
| Regresión logística | 0.7435 | 0.0169 |

La comparación pareada mediante Wilcoxon obtuvo:

```text
p = 0.001953125
```

Por tanto, la diferencia entre el MLP multimodal y la regresión logística fue estadísticamente significativa con α = 0.05.

Evaluación final del MLP optimizado:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7578 |
| Precision | 0.7785 |
| Recall | 0.8467 |
| F1-score | 0.8112 |
| ROC-AUC | 0.8411 |
| PR-AUC | 0.8982 |

## Ejecución en Google Colab

1. Abrir el notebook en Google Colab.
2. Seleccionar **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.
3. Verificar la GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

4. Instalar dependencias:

```python
!pip install -q transformers datasets huggingface_hub optuna seaborn scipy
```

5. Ejecutar las celdas en orden, una por una.

El notebook fue ejecutado usando una GPU Tesla T4 de Google Colab.

## Dependencias principales

- Python 3
- PyTorch
- Transformers
- Hugging Face Hub
- NumPy
- pandas
- scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Optuna
- Pillow

## Estructura recomendada

```text
.
├── CrisisMMD_Multimodal_Jessica_Ballesteros.ipynb
├── README.md
└── data/                 # creada durante la ejecución
```

## Feature selection

No se aplicó una técnica adicional de selección de características porque el proyecto utiliza TF-IDF con un vocabulario limitado y embeddings visuales CLIP obtenidos mediante transferencia de aprendizaje. Esta decisión está permitida por la rúbrica para problemas de texto e imágenes con transfer learning.

## Limitaciones

- Se utilizó una muestra del 10 % para reducir el tiempo de ejecución en Colab.
- CLIP se utilizó congelado, sin fine-tuning.
- El dataset contiene publicaciones en inglés.
- Algunas clases pueden presentar patrones específicos de cada evento.
- La evaluación final corresponde a la partición oficial `test` de la muestra utilizada.

## Trabajo futuro

- Ejecutar el experimento con el dataset completo.
- Aplicar fine-tuning parcial de CLIP.
- Utilizar modelos de lenguaje como BERT o RoBERTa.
- Incorporar calibración probabilística.
- Evaluar un split temporal por evento.
- Probar mecanismos de umbral adaptativo.

## Uso de inteligencia artificial

Se utilizaron herramientas de inteligencia artificial como apoyo para comprender conceptos, estructurar código, interpretar resultados y mejorar la redacción. Los prompts utilizados se documentan dentro del notebook. La selección metodológica, ejecución, revisión de resultados y conclusiones fueron realizadas para este proyecto.

## Licencia y fuente de datos

El dataset pertenece a sus autores y se encuentra disponible públicamente en Hugging Face:

[QCRI/CrisisMMD](https://huggingface.co/datasets/QCRI/CrisisMMD)
