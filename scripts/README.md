# Scripts

This folder contains the scripts to reproduce the audit end to end.

## Pipeline overview

1. `00_prep_metadata.py` reads the CCv2 annotations JSON and produces two CSV files in `configs/`: the subject-level metadata (`ccv2_metadata.csv`) and the frame manifest (`frame_manifest.csv`).

2. `01_run_detection.py` runs one detector against every frame listed in the manifest and writes a CSV in `results/detections/` with one row per frame. Supported detectors: haar, mtcnn, retinaface, scrfd, yolo11_ft, yolo12, rtdetr. Optional flags: `--weights` for a custom checkpoint, `--output-name` for a custom output filename.

3. `02_analyze_coverage.py` computes stratified coverage (global, univariate, bivariate, trivariate) from a detections CSV.

4. `03_overlap_analysis.py` computes the overlap of missed frames between detectors and the demographic profile of architecture-exclusive misses.

5. `04_train_scratch.py` trains YOLOv11, YOLOv12 or RT-DETR from scratch on CCv2, using pseudo-labels from RetinaFace as supervision.

## Minimum reproduction

```
python 00_prep_metadata.py
python 01_run_detection.py --detector retinaface --conf 0.5
python 01_run_detection.py --detector yolo11_ft --conf 0.25
python 02_analyze_coverage.py
python 03_overlap_analysis.py
```

---

# Scripts (português)

Esta pasta contém os scripts para reproduzir a auditoria do início ao fim.

## Visão geral do pipeline

1. `00_prep_metadata.py` lê o JSON de anotações do CCv2 e gera dois CSVs em `configs/`: metadados por participante (`ccv2_metadata.csv`) e manifesto de frames (`frame_manifest.csv`).

2. `01_run_detection.py` roda um detector em cada frame do manifesto e escreve um CSV em `results/detections/`. Detectores suportados: haar, mtcnn, retinaface, scrfd, yolo11_ft, yolo12, rtdetr. Flags opcionais: `--weights` para checkpoint customizado, `--output-name` para nome de saída customizado.

3. `02_analyze_coverage.py` calcula cobertura estratificada (global, univariada, bivariada, trivariada) a partir de um CSV de detecções.

4. `03_overlap_analysis.py` calcula a sobreposição de frames perdidos entre detectores e o perfil demográfico do apagamento exclusivo de arquitetura.

5. `04_train_scratch.py` treina YOLOv11, YOLOv12 ou RT-DETR do zero no CCv2, usando pseudo-labels do RetinaFace como supervisão.

## Reprodução mínima

```
python 00_prep_metadata.py
python 01_run_detection.py --detector retinaface --conf 0.5
python 01_run_detection.py --detector yolo11_ft --conf 0.25
python 02_analyze_coverage.py
python 03_overlap_analysis.py
```
