# Coverage Before Recognition: Auditing Architectural Erasure in Face Detection

Published at BRACIS 2026, AI for Social Good Track. Lecture Notes in Artificial Intelligence, volume 17109, Springer, pages 608 to 623.

**Authors:** Nina da Hora (1), Marisol Marini (2), Sandra Avila (1)
(1) Institute of Computing, University of Campinas (UNICAMP)
(2) University of São Paulo (USP)

**Paper:** https://doi.org/10.1007/978-3-032-39904-5_41

*Versão em português abaixo.*

---

## Overview

Face detection is the first step in facial analysis pipelines. If detection fails, no downstream stage runs: the person does not receive a wrong label or an inaccurate match, the person simply does not exist for the system. Yet fairness audits focus on later stages, such as recognition and classification, leaving the question of who the detector fails to see unexamined.

This repository contains the code, data processing scripts, and analysis notebooks for an audit of seven face detectors across six architectural paradigms, evaluated on the full Casual Conversations v2 dataset (196,397 frames, 4,047 subjects, 7 countries).

## Key Findings

| Finding | Value |
|---|---|
| Global coverage of YOLOv11 (CNN pre-trained on COCO) | 93.22% |
| Coverage for Monk 9 subjects | 74.00% |
| Coverage for cis women, Monk 9, age 56 to 65 | 6.67% |
| Frames missed exclusively by YOLOv11, detected by all other systems | 10,258 |
| Coverage of YOLOv11 trained from scratch (no COCO) at Monk 9 | 97.75% |

The visibility gradient disappears when YOLOv11 is trained from scratch on CCv2 without COCO pre-training. The disparity arises from the interaction between the CNN architecture and its pre-training heritage, not from the CNN paradigm alone. Attention-based architectures (YOLOv12, RT-DETR) process the same COCO data without producing the gradient.

## Repository Structure

```
.
├── paper/              Paper PDF and LaTeX sources
├── configs/            CCv2 metadata and frame manifest
├── scripts/            Detection and training scripts
├── results/            CSV outputs with detection results per detector
├── notebooks/          Analysis notebooks reproducing the paper tables
├── figures/            Figures used in the paper
├── requirements.txt    Python dependencies
└── LICENSE             MIT License
```

## Requirements

- Python 3.10 or higher
- PyTorch 2.11 or higher
- Ultralytics 8.3 or higher
- InsightFace 0.7.3
- facenet-pytorch 2.5.3
- opencv-python, pandas, numpy, scipy, matplotlib, seaborn

Install with:

```
pip install -r requirements.txt
```

## Dataset

The Casual Conversations v2 dataset is distributed by Meta under a research license. Access it at https://ai.meta.com/datasets/casual-conversations-v2-dataset/. This repository does not redistribute the dataset.

## Reproducing the Results

### Step 1. Prepare metadata and frame manifest

```
python scripts/00_prep_metadata.py
```

### Step 2. Run detection for the seven detectors

```
python scripts/01_run_detection.py --detector retinaface --conf 0.5
python scripts/01_run_detection.py --detector yolo11_ft --conf 0.25
python scripts/01_run_detection.py --detector yolo12 --conf 0.25
python scripts/01_run_detection.py --detector rtdetr --conf 0.3
python scripts/01_run_detection.py --detector scrfd --conf 0.5
python scripts/01_run_detection.py --detector mtcnn --conf 0.5
python scripts/01_run_detection.py --detector haar --conf 0.0
```

### Step 3. Train the scratch models

```
python scripts/04_train_scratch.py --model yolo11n --epochs 100
python scripts/04_train_scratch.py --model yolo12n --epochs 100
python scripts/04_train_scratch.py --model rtdetr-l --epochs 80
```

### Step 4. Analyze stratified coverage

```
python scripts/02_analyze_coverage.py
python scripts/03_overlap_analysis.py
```

## Citation

```bibtex
@inproceedings{dahora2026coverage,
  title     = {Coverage Before Recognition: Auditing Architectural Erasure in Face Detection},
  author    = {da Hora, Nina and Marini, Marisol and Avila, Sandra},
  booktitle = {Brazilian Conference on Intelligent Systems (BRACIS)},
  series    = {Lecture Notes in Artificial Intelligence},
  volume    = {17109},
  pages     = {608--623},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-39904-5_41}
}
```

## Contact

- Nina da Hora: ninadhoraa@gmail.com

## License

Code: MIT License, see LICENSE file.
Dataset usage: governed by Meta's Casual Conversations v2 license.

---

# Versão em português

## Visão geral

A detecção facial é a primeira etapa de qualquer pipeline de análise facial. Se a detecção falha, nenhuma etapa seguinte roda: a pessoa não recebe um rótulo errado nem um pareamento incorreto, ela simplesmente não existe para o sistema. Mesmo assim, auditorias de fairness se concentram nas etapas posteriores, como reconhecimento e classificação, deixando a pergunta sobre quem o detector não vê sem resposta.

Este repositório contém o código, scripts de processamento de dados e notebooks de análise de uma auditoria de sete detectores faciais em seis paradigmas arquiteturais, avaliados no conjunto completo do Casual Conversations v2 (196.397 frames, 4.047 participantes, 7 países).

## Principais achados

| Achado | Valor |
|---|---|
| Cobertura global do YOLOv11 (CNN pré-treinada no COCO) | 93,22% |
| Cobertura para participantes com Monk 9 | 74,00% |
| Cobertura para mulheres cis, Monk 9, idade 56 a 65 | 6,67% |
| Frames perdidos somente pelo YOLOv11, detectados por todos os outros | 10.258 |
| Cobertura do YOLOv11 treinado do zero (sem COCO) em Monk 9 | 97,75% |

O gradiente de visibilidade desaparece quando o YOLOv11 é treinado do zero no CCv2 sem o pré-treino COCO. A disparidade surge da interação entre a arquitetura CNN e sua herança de pré-treino, não da arquitetura CNN sozinha. Arquiteturas com atenção (YOLOv12, RT-DETR) processam os mesmos dados do COCO sem produzir o gradiente.

## Estrutura do repositório

```
.
├── paper/              PDF do artigo e código-fonte LaTeX
├── configs/            Metadados do CCv2 e manifesto de frames
├── scripts/            Scripts de detecção e treino
├── results/            CSVs com resultados de detecção por detector
├── notebooks/          Notebooks de análise que reproduzem as tabelas
├── figures/            Figuras usadas no artigo
├── requirements.txt    Dependências Python
└── LICENSE             Licença MIT
```

## Requisitos

- Python 3.10 ou superior
- PyTorch 2.11 ou superior
- Ultralytics 8.3 ou superior
- InsightFace 0.7.3
- facenet-pytorch 2.5.3
- opencv-python, pandas, numpy, scipy, matplotlib, seaborn

Instalação:

```
pip install -r requirements.txt
```

## Dataset

O Casual Conversations v2 é distribuído pela Meta sob licença de pesquisa. Acesso em https://ai.meta.com/datasets/casual-conversations-v2-dataset/. Este repositório não redistribui o dataset.

## Reproduzindo os resultados

### Passo 1. Preparar metadados e manifesto de frames

```
python scripts/00_prep_metadata.py
```

### Passo 2. Rodar detecção para os sete detectores

```
python scripts/01_run_detection.py --detector retinaface --conf 0.5
python scripts/01_run_detection.py --detector yolo11_ft --conf 0.25
python scripts/01_run_detection.py --detector yolo12 --conf 0.25
python scripts/01_run_detection.py --detector rtdetr --conf 0.3
python scripts/01_run_detection.py --detector scrfd --conf 0.5
python scripts/01_run_detection.py --detector mtcnn --conf 0.5
python scripts/01_run_detection.py --detector haar --conf 0.0
```

### Passo 3. Treinar os modelos do zero

```
python scripts/04_train_scratch.py --model yolo11n --epochs 100
python scripts/04_train_scratch.py --model yolo12n --epochs 100
python scripts/04_train_scratch.py --model rtdetr-l --epochs 80
```

### Passo 4. Analisar cobertura estratificada

```
python scripts/02_analyze_coverage.py
python scripts/03_overlap_analysis.py
```

## Como citar

```bibtex
@inproceedings{dahora2026coverage,
  title     = {Coverage Before Recognition: Auditing Architectural Erasure in Face Detection},
  author    = {da Hora, Nina and Marini, Marisol and Avila, Sandra},
  booktitle = {Brazilian Conference on Intelligent Systems (BRACIS)},
  series    = {Lecture Notes in Artificial Intelligence},
  volume    = {17109},
  pages     = {608--623},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-39904-5_41}
}
```

## Contato

- Nina da Hora: ninadhoraa@gmail.com
## Licença

Código: Licença MIT, ver arquivo LICENSE.
Uso do dataset: regido pela licença do Casual Conversations v2 da Meta.
