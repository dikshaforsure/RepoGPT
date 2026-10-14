# RepoGPT — Gemini + local embeddings

A Streamlit app for asking questions about a GitHub repository. It reads supported
source files from GitHub, embeds them locally with Hugging Face, retrieves relevant
chunks with LlamaIndex, and uses the Gemini API to generate answers.

## Requirements

- Python 3.11 or 3.12 recommended
- Git
- A Gemini API key from https://aistudio.google.com/api-keys
- A GitHub fine-grained personal access token with read access to public repositories

Gemini API usage depends on the model's current free-tier limits and quotas. The
embedding model runs locally and does not need an embedding API key. The first run
downloads the embedding model, so allow time and disk space for that download.

## Windows PowerShell setup

```powershell
git clone https://github.com/dikshaforsure/RepoGPT.git
cd RepoGPT

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env
```

Add your keys to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
GITHUB_TOKEN=your_github_token
```

Save the file, then run:

```powershell
streamlit run main.py
```

Open http://localhost:8501. You can also enter the keys in the sidebar instead of
putting them in `.env`.

If PowerShell blocks virtual-environment activation, run this only in the current
terminal and retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Notes

- This version does not use a Nebius API key.
- The Gemini API generates answers; Hugging Face embeddings run locally.
- The vector index is held in memory and is lost when the Python process stops.
- Indexing a large repository may take time and local RAM.
- Do not commit `.env` or expose API tokens.
