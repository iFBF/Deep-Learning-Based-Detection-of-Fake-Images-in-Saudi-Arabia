# Real-Abshar V3.1 local comparison portal

This add-on gives the existing V3.1 research policy a browser interface. It puts the **combined review recommendation** first, then shows Swin, the supplemental generation model, edited-region boxes and the segmentation mask separately. An unflagged result is labelled **No model flag**, never an authenticity certificate.

It uses your existing V3.1 Python environment and the original M8 service. No model download, training, Kaggle run or new package installation is needed if the comparison client already works.

## What your pasted result means

- `review_recommended: true`: the combined system recommends human review.
- `swin_generation: false`: Swin did not cross the selected 0.99 cutoff.
- `supplemental_generation: true`: the supplemental raw logit crossed 4.25.
- `edited_region_box: true` and `edited_area: true`: both original local checks flagged.
- `authenticity_verified: false`: the models do not certify origin or authenticity.
- `joint_service_validation_complete: false`: V3.1 cohort quality was recomputed from cached validation outputs. This single-image check is not a new formal live cohort evaluation. This field is not a prediction error.

The snippet establishes that your comparison client returned a combined decision. The full JSON is still needed to inspect its release ID, exact score and timings. It does not by itself establish accuracy on new images.

## Start the browser portal on Windows

Keep the original service already running on **http://127.0.0.1:8000**. Do not launch a duplicate original server. The new browser portal uses **http://127.0.0.1:8001**.

Download **Real-Abshar_V3_1_Portal_Addon.zip** into Downloads. In a second PowerShell window, paste the entire block:

```powershell
$ErrorActionPreference = 'Stop'
$researchRoot = Join-Path $env:USERPROFILE 'Real-Abshar-V3-1-Research'
$portalPython = Join-Path $researchRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $portalPython -PathType Leaf)) {
    throw 'The working V3.1 Python environment was not found in Real-Abshar-V3-1-Research.'
}
$baselineHealth = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'
if ($baselineHealth.status -ne 'ready' -or $baselineHealth.release_id -ne 'real-abshar-m8-fbaa1c95818d') {
    throw 'Port 8000 does not contain the expected ready original service.'
}
$portalZip = Get-ChildItem -LiteralPath "$env:USERPROFILE\Downloads" -Filter 'Real-Abshar_V3_1_Portal_Addon*.zip' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $portalZip) { throw 'Download Real-Abshar_V3_1_Portal_Addon.zip into Downloads first.' }
$portalRoot = Join-Path $env:USERPROFILE 'Real-Abshar-V3-1-Portal'
if (Test-Path -LiteralPath $portalRoot) {
    throw 'The portal folder already exists. Use the restart command below instead of extracting again.'
}
New-Item -ItemType Directory -Path $portalRoot | Out-Null
Expand-Archive -LiteralPath $portalZip.FullName -DestinationPath $portalRoot
& $portalPython -X utf8 (Join-Path $portalRoot 'portal_server.py') --release $researchRoot --baseline-url 'http://127.0.0.1:8000' --port 8001 --open-browser
```

Wait for **V3.1 local research portal ready**. Your browser should open automatically. Otherwise open **http://127.0.0.1:8001** yourself. Keep both PowerShell service windows open.

### Restart after closing the new portal

With the original service running on port 8000, paste:

```powershell
& "$env:USERPROFILE\Real-Abshar-V3-1-Research\.venv\Scripts\python.exe" -X utf8 "$env:USERPROFILE\Real-Abshar-V3-1-Portal\portal_server.py" --release "$env:USERPROFILE\Real-Abshar-V3-1-Research" --baseline-url 'http://127.0.0.1:8000' --port 8001 --open-browser
```

If port 8001 is occupied, open its `/health` address. If it is this portal, reuse it. If another application occupies it, change **only `--port 8001` to `--port 8002`** and open the corresponding address. Keep `--baseline-url` pointing to port 8000. There is no need to stop an unidentified process.

## Use and save one result

1. Choose a JPEG, PNG or WebP image, up to 10 MiB. The existing model contract also limits inputs to 12 million pixels, minimum side 16 pixels and a single still frame.
2. Click **Analyze image**. The headline is the combined four-model recommendation.
3. Inspect component flags. The original Swin label may still say Real while the combined system recommends review; the Swin label applies only to that component at its original cutoff.
4. Toggle **Region boxes** and **Pixel mask** to inspect local predictions. Mask coverage is predicted foreground, not a verified generated fraction. Blue boxes and the pink mask use native oriented-image coordinates from the original service; resizing is for display only.
5. Click **Download result JSON**. Send **latest_v31_portal_comparison.json** for review. Save the earlier latest_v31_comparison.json too if available.

The add-on itself does not automatically save photos or result files. Its browser download saves the JSON only when requested. A mask is included as base64 PNG; the original image bytes are not embedded in that JSON. Requests go to the original loopback service, and the portal listens on loopback only.

## Policy, provenance and verification

The V3.1 candidate is read and checksum-verified without modifying its payload. Swin uses 224×224 and score cutoff 0.99; the supplemental model uses 384×384 and raw cutoff 4.25. Full referral adds unchanged original M4/M5 flags: 640×640 inputs, detector confidence 0.35, NMS IoU 0.45, maximum 50 boxes, mask pixel cutoff 0.5 and minimum foreground fraction 0.005. The supplemental score is a raw logit, not a verified probability.

No new performance evaluation is performed by this add-on. Its per-request processing time is not a benchmark or a newly measured p95. V3.1 remains a validation-selected research candidate. Original closed tests remain closed; this portal does not change GitHub, W&B or the grader submission. Full provenance, external model credit and numerical validation remain in the existing candidate's MODEL_CARD.md and JOINT_VALIDATION_REVIEW.md. The externally trained supplemental checkpoint must not be presented as trained by the team.

The connection layer was tested with actual supplemental ONNX inference and a synthetic original-service HTTP fixture. Browser QA used clearly labelled synthetic responses for flagged/unflagged results, overlays and the download action. The private M8 models and Windows installation were not available in that QA environment. Your pasted output is separate user-reported evidence of the Windows comparison client. See QA_REPORT.json for the exact verification scope.

Keep the original V1 service available for the course presentation and identify this add-on as a separate V3.1 research demonstration. Automatic authenticity decisions remain NO-GO.
