# Results

This folder stores the CSV outputs produced by `scripts/01_run_detection.py`.
Each CSV has one row per frame with columns:

- `frame_path`: absolute path to the input frame.
- `subject_id`: four-digit CCv2 participant ID.
- `n_detections`: number of boxes returned by the detector for that frame.
- `max_conf`: highest confidence among the returned boxes.
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2`: coordinates of the top-confidence box.
- `detected`: True if at least one box was returned, False otherwise.

A summary of the global coverage reported in the paper is in `global_coverage.csv`.

---

# Resultados (português)

Esta pasta guarda os CSVs produzidos por `scripts/01_run_detection.py`. Cada
CSV tem uma linha por frame com as colunas:

- `frame_path`: caminho absoluto do frame de entrada.
- `subject_id`: ID de quatro dígitos do participante no CCv2.
- `n_detections`: número de caixas retornadas pelo detector naquele frame.
- `max_conf`: maior confiança entre as caixas retornadas.
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2`: coordenadas da caixa de maior confiança.
- `detected`: True se ao menos uma caixa foi retornada, False caso contrário.

Um resumo da cobertura global reportada no artigo está em `global_coverage.csv`.
