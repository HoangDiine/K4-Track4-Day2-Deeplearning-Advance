# DeepWeeds Lab — submission package

## Contents

- `results.xlsx`: comparison tables and saved `eval.py` scores.
- `report.md`: results, analysis, limits, and recommendation.
- `curves/`: training curves for every recorded B/T/F run.
- `code/`: source modules and evaluation code.
- `predictions/`: final and validation CSVs for seeds 0–2.
- `raw_results/`: compact JSON/CSV source records; no model weights.
- `lab_day2.ipynb`: executed Colab notebook used for this run.

## Re-run

Open the notebook in [Google Colab](https://colab.research.google.com/github/HoangDiine/K4-Track4-Day2-Deeplearning-Advance/blob/main/starter/lab_day2.ipynb), select a GPU runtime, and run cells top to bottom. The notebook mounts Drive and stores generated runs there. To reproduce this exact local revision, upload the included notebook and `code/` folder to Colab (the public branch may not yet contain local fixes). Python package versions and the recorded hardware are in the notebook and workbook. The saved run used seed 0 for screening and seeds 0, 1, 2 for the final/baseline.

For local scoring of saved predictions, from the repository root run `python eval.py score` and `python eval.py grade`; see the notebook's scoring cells for complete arguments.

The final test outputs used I00. I01 was evaluated on validation only and must not be presented as a test result.
