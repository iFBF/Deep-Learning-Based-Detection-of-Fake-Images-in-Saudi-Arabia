# Run the evaluated service

The working service is a local FastAPI application. Public hosting is not claimed. Keep the private `Real-Abshar_Phase2_M8_Service_Release.zip` available through the final presentation. Give graders the release through an authorized private channel; no dataset is needed to run the service on a user-supplied image.

From this repository root, use Python 3.13 and a fresh virtual environment. Linux/macOS:

```bash
python3.13 -m venv .venv-phase2
.venv-phase2/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.venv-phase2/bin/python -m pip install -r phase2/service/requirements_serve.txt
.venv-phase2/bin/python phase2/restore_service_models.py --archive /absolute/path/Real-Abshar_Phase2_M8_Service_Release.zip
.venv-phase2/bin/python phase2/service/serve.py --release phase2/service --port 8000
```

Windows PowerShell (use the actual private archive path):

```powershell
py -3.13 -m venv .venv-phase2
.\.venv-phase2\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-phase2\Scripts\python.exe -m pip install -r phase2/service/requirements_serve.txt
.\.venv-phase2\Scripts\python.exe phase2/restore_service_models.py --archive "C:\path\Real-Abshar_Phase2_M8_Service_Release.zip"
.\.venv-phase2\Scripts\python.exe phase2/service/serve.py --release phase2/service --port 8000
```

Open http://127.0.0.1:8000/ for the image form, http://127.0.0.1:8000/health for health, or http://127.0.0.1:8000/docs for the API. Upload a still JPEG, PNG or WebP; all three branches run. A no-model-flag result is not an authenticity certificate. Stop with Ctrl+C.

The benchmark and HTTP contract tests were actually run on Kaggle Linux CPU; Windows commands are provided for reproduction and were not benchmarked here. The frozen `service/RUN_LOCAL.md` has additional contract/curl/timer detail. The restoration helper copies only the three hash-checked graphs. No W&B login is needed after setup and no training is performed.

The copied source/configuration hashes and three model hashes are checked at startup. Preserve the historical frozen manifest and lock unchanged. Final evaluation completion is recorded separately in `results/completed_evaluation.json` and the final model card.
