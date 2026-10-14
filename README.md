# RepoGPT — Groq + local embeddings

A Streamlit app for asking questions about a GitHub repository. It reads supported
source files from GitHub, embeds them locally with Hugging Face, retrieves relevant
chunks with LlamaIndex, and uses the Groq API to generate answers.

## Requirements

- Python 3.11 or 3.12 recommended
- Git
- A Groq API key from https://console.groq.com/keys
- A GitHub personal access token with read access to the repository

Groq API access is subject to the model and rate limits for your account. The
embedding model runs locally and does not need an embedding API key. The first run
downloads the embedding model, so allow time and disk space for that download.

## Windows setup (Command Prompt)

```bat
git clone https://github.com/dikshaforsure/RepoGPT.git
cd RepoGPT

py -3.12 -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

copy .env.example .env
notepad .env
```

Add your keys to `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GITHUB_TOKEN=your_github_token
GROQ_MODEL=openai/gpt-oss-120b
EMBED_MODEL=BAAI/bge-small-en-v1.5
```

Save the file, then start the app:

```bat
python -m streamlit run main.py --server.port 8503 --server.fileWatcherType none
```

Open the local URL printed by Streamlit (normally http://localhost:8503). You can
also enter the Groq API key and GitHub token in the app's sidebar instead of
putting them in `.env`.

If you use PowerShell instead of Command Prompt, activate the environment with
`\.venv\Scripts\Activate.ps1`. If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal
and retry activation.

## Notes

- Groq generates answers; Hugging Face embeddings run locally.
- This project does not require a Gemini or Nebius API key.
- The vector index is held in memory and is lost when the Python process stops.
- Indexing a large repository may take time and local RAM.
- If Groq reports that a model is unavailable for your account, choose a supported
  model from https://console.groq.com/docs/models and update `GROQ_MODEL` in `.env`.
- Do not commit `.env` or expose API tokens.
