# Run the V3.1 research portal

Use Python3.13. Keep the original M8 service and its verified private graphs available through the final presentation. The original run instructions are ../../RUN_LOCAL.md. On Windows launch the original service with UTF-8 enabled:

```powershell
python -X utf8 phase2/service/serve.py --release phase2/service --port 8000
```

If http://127.0.0.1:8000/health already shows ready and `real-abshar-m8-fbaa1c95818d`, reuse it. The new portal uses8001.

## Existing working Windows installation

The validated user installation stores the canonical V3.1 candidate under `$env:USERPROFILE\Real-Abshar-V3-1-Research`, with its own `.venv`. From this repository root, run:

```powershell
& "$env:USERPROFILE\Real-Abshar-V3-1-Research\.venv\Scripts\python.exe" -X utf8 .\phase2\research\v31\portal\portal_server.py --release "$env:USERPROFILE\Real-Abshar-V3-1-Research" --baseline-url 'http://127.0.0.1:8000' --port 8001 --open-browser
```

Upload a supported still image, click Analyze image and download the JSON if needed. Keep both service windows open. Ctrl+C in the new portal window stops only that portal. If another application occupies8001, use `--port 8002`; keep the original baseline URL8000.

## Fresh authorized installation

Obtain `Real-Abshar_V3_1_Research_Candidate.zip` from the team/private course archive. Canonical ZIP SHA256: `4e959d9815c7bfe853f060913c5f975cbeceeb650ba6cecb47444b8936d04dd8` (81,589,868bytes). The candidate manifest and individual payload hashes are checked by the runtime. The original M8 service release is a separate prerequisite; no CSI dataset is needed for ordinary image inference.

Extract the candidate outside the code repository. Create a separate environment and install the candidate's requirements_light.txt (NumPy2.1.3,Pillow12.3.0,ORT1.31.1). Run the portal above with `--release` set to that extracted candidate folder and its Python interpreter. No PyTorch or new training is needed in the V3.1 client environment. The original service has its separate existing dependencies.

For a command-line comparison, run `runtime/predict_v3.py --release PATH_TO_PRIVATE_CANDIDATE --image IMAGE_PATH --baseline-url http://127.0.0.1:8000 --output RESULT.json` using the same environment. For native development reproduction or the optional live CSI replay, run the scripts/notebook inside the complete private candidate; their private input/fingerprint records are intentionally not duplicated in this public snapshot.

The combined decision is review_recommended or no_model_flag. A component's Real label is not the system's overall conclusion. Binary mask coverage is predicted foreground, not a verified edited fraction. `joint_service_validation_complete:false` preserves the current cached-validation status; a single image does not establish a cohort evaluation. Original benchmark p95 must not be replaced with one portal-request time.
