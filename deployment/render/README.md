# Teacher Support Studio — Render bundle

This folder is a self-contained, reviewer-facing deployment of Teacher Support
Studio. The full local application, notebooks, datasets, and development
environment remain unchanged elsewhere in the repository.

Files under `src/`, `data/`, `models/`, and `outputs/` in this folder are
generated deployment copies. Do not edit them directly. Refresh them from the
project root with:

```powershell
.\.venv\Scripts\python.exe scripts\build_render_bundle.py
```

## Deploy on Render

1. Push the repository, including this generated bundle, to GitHub.
2. In Render, create a **Blueprint** and connect the GitHub repository.
3. If Render asks for the Blueprint path, enter
   `deployment/render/render.yaml`.
4. Confirm that the service uses the **Free** instance type and create it.
5. Wait for `/api/health` to pass, then open the assigned public URL.

The application works without an OpenAI key by returning deterministic guided
responses. To enable live model-generated responses, add `OPENAI_API_KEY` as a
secret environment variable in the Render dashboard. Never put the key in this
folder or commit it to Git.

Render builds only this directory. The deployment does not require the full CSV
files stored with Git LFS.
