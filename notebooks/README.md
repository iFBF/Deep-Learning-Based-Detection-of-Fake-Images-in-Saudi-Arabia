# Phase 1 notebooks

Use `phase1_independent_reproduction.ipynb` for the required actual independent member run. It installs the pinned dependencies and executes data preparation, fresh training, validation-only selection, fixed final evaluation and exports in a fresh Python process. Enter the actual operator's name and use a fresh T4 GPU runtime with code and original data inputs attached.

`phase1_corrections_kaggle.ipynb` is the clean main-workflow notebook. Its data cells use the fixed development manifest. The completed main run and the review record are in `../docs/phase1_corrections/completed_run/`.

The earlier executed notebook is preserved as historical exploratory evidence; its old test score is not the corrected result. Independent member execution of the new notebook is still pending.
