# Hosting the PE Finance Dashboard

The dashboard is a **Streamlit** multi-page app. The entry point is `dashboard.py` at the project root; additional tools live in `pages/`.

## Local development

```bash
uv venv
uv pip install -e ".[dashboard]"
streamlit run dashboard.py
```

Open http://localhost:8501.

## Free cloud hosting options

| Platform | Why it works | Steps |
|----------|--------------|-------|
| **Streamlit Community Cloud** | Built for Streamlit, free tier | Connect GitHub repo, choose `dashboard.py`, deploy. Set `PYTHON_VERSION` to 3.11. |
| **Hugging Face Spaces** | Free, supports Streamlit Docker template | Create a new Space with the Streamlit SDK, push code, set `app_file: dashboard.py` in `README.md`. |
| **Render** | Free web services, easy Docker support | Create a Web Service, connect repo, build command `uv pip install -e ".[dashboard]"`, start command `streamlit run dashboard.py --server.port 10000`. |

## Notes

- The app only needs Python packages from `pyproject.toml` (no separate `requirements.txt` required by Streamlit Cloud or Render, but you can add one if you prefer).
- For public hosting, upload and regenerate functions use the container's filesystem, so uploaded files do not persist across sessions unless you add cloud storage.
- Use `python` and `streamlit` versions compatible with `pyproject.toml` (`>=3.11`, `streamlit>=1.35`).
