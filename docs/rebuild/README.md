# Sign Avatar rebuild

A browser-based NSL research application being rebuilt to cover data preparation, models, user-facing flows and validation. Stage 1 provides the browser interface and local API foundation. This project is evidence of technical experience for a potential MANAD collaboration; it does not claim to implement MSL.

## Run stage 1 locally

Use the rebuild/sign-avatar branch. Node 22+ and Python 3.11+ are required. Keep the recovered worktrees alongside their parent repository; no recovered file is moved or rewritten by this stage.

From the repository root:

```bash
python3 -m venv .venv-web
source .venv-web/bin/activate
python -m pip install -e './backend[test]'
npm --prefix web ci
npm --prefix web run build
python -m uvicorn sign_avatar_api.app:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. Run commands using the root of the checkout, not backend/. The editable installation intentionally uses the shared catalogue and build output from the checkout.

For UI development, leave the API running and run `npm --prefix web run dev` in a second terminal. Vite proxies /api to port 8000. Use the URL Vite prints, normally http://localhost:5173.

The API binds to the local machine. Phone/tablet access, authentication and HTTPS are a later server-deployment stage. Camera capture works on localhost or HTTPS; simply binding to a LAN address does not provide a secure browser context.

## What works in this stage

- Responsive Translate, Sign library, Dataset, Experiments and Settings screens.
- Camera permission, preview, isolated clip recording (maximum 15 seconds), stop and close, playback and local download. No audio is recorded.
- Selection and playback of a local video up to 100 MB. No upload occurs.
- Search/category filtering of the canonical 49 recovered model labels.
- Exact vocabulary lookup with unsupported tokens visible and repeated tokens preserved. This is not sentence translation or NSL grammar processing.
- Light/dark appearance stored only in the current browser.
- Local health and catalogue API; optional serving of the production UI.
- Historical recovery information clearly separated from current measurements.

Recognition, model installation, skeletal playback, speech processing, dataset persistence/import, training and evaluation execution are not implemented here. Their controls are unavailable rather than returning invented results. The UI bundles the catalogue for browsing even when the backend is offline.

## Verification

```bash
npm --prefix web run build
python -m pytest backend/tests -q
```

For manual device testing: allow/deny camera permission, record/stop, close camera during recording, navigate away during capture, download/replay a clip, choose unsupported video formats, test at narrow widths and with keyboard navigation. Camera tracks must stop when leaving capture. This stage has no automated physical-camera validation.

See [Product and workflows](product-and-workflows.md) and [Implementation stages](stages.md).

Optional WebMCP library navigation is feature-detected. It has not been validated in a supported browser context; ordinary navigation does not depend on it.
