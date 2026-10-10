"""
Codebase Q&A RAG — chat with any public GitHub repository.

Pipeline: GitHub repo -> LlamaIndex document loader -> Nebius embeddings ->
in-memory vector index -> top-k retrieval -> DeepSeek-V3 answer synthesis.
"""

import os
import re
import time

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.embeddings.nebius import NebiusEmbedding
from llama_index.llms.nebius import NebiusLLM
from llama_index.readers.github import GithubClient, GithubRepositoryReader

# GithubRepositoryReader drives asyncio internally; Streamlit already owns an
# event loop on the script thread, so nesting has to be allowed explicitly.
nest_asyncio.apply()
load_dotenv()

LLM_MODEL = "deepseek-ai/DeepSeek-V3"
EMBED_MODEL = "BAAI/bge-en-icl"

FILE_EXTENSIONS = [
    ".py", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt",
    ".sql", ".sh", ".md", ".rst", ".toml", ".yaml", ".yml", ".json",
]

QA_PROMPT = PromptTemplate(
    "You are a senior engineer helping someone understand an unfamiliar codebase.\n"
    "Below are excerpts retrieved from the repository.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer the question using only the context above. Cite the file paths you\n"
    "relied on. Show short code snippets when they make the answer concrete. If\n"
    "the context does not contain the answer, say so plainly instead of guessing.\n\n"
    "Question: {query_str}\n"
    "Answer: "
)


def parse_github_url(url: str) -> tuple[str, str, str]:
    """Split a GitHub URL into (owner, repo, branch). Defaults to the 'main' branch."""
    pattern = r"https?://(?:www\.)?github\.com/([^/]+)/([^/\s]+?)(?:\.git)?/?(?:/tree/([^/\s]+))?$"
    match = re.match(pattern, url.strip())
    if not match:
        raise ValueError(
            "Invalid GitHub URL. Expected https://github.com/<owner>/<repo> "
            "or https://github.com/<owner>/<repo>/tree/<branch>"
        )
    owner, repo, branch = match.groups()
    return owner, repo, branch or "main"


@st.cache_resource(show_spinner=False)
def load_repository(github_token: str, owner: str, repo: str, branch: str):
    """Download and parse every supported source file in the repo. Cached per repo+branch."""
    loader = GithubRepositoryReader(
        GithubClient(github_token, verbose=False),
        owner=owner,
        repo=repo,
        filter_file_extensions=(FILE_EXTENSIONS, GithubRepositoryReader.FilterType.INCLUDE),
        verbose=False,
        concurrent_requests=5,
    )
    return loader.load_data(branch=branch)


@st.cache_resource(show_spinner=False)
def build_index(_docs, nebius_api_key: str, cache_key: str) -> VectorStoreIndex:
    """Embed the documents once and keep the vector index alive across reruns.

    The index is the expensive artifact — rebuilding it per question would re-embed
    the whole repository every time, so it is cached on repo identity (`cache_key`).
    """
    configure_models(nebius_api_key)
    return VectorStoreIndex.from_documents(_docs, show_progress=False)


def configure_models(nebius_api_key: str) -> None:
    Settings.llm = NebiusLLM(model=LLM_MODEL, api_key=nebius_api_key, temperature=0.1)
    Settings.embed_model = NebiusEmbedding(model_name=EMBED_MODEL, api_key=nebius_api_key)


def get_secret(session_value: str, env_var: str) -> str:
    """Prefer a key typed into the sidebar, fall back to the environment / Space secrets."""
    return (session_value or os.getenv(env_var) or "").strip()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Setup")

        st.text_input(
            "Nebius API key",
            type="password",
            key="nebius_key",
            help="Free credits at studio.nebius.ai — powers both embeddings and the LLM.",
        )
        st.text_input(
            "GitHub token",
            type="password",
            key="github_key",
            help="A fine-grained token with public-repo read access. Raises your rate limit "
            "and lets you index private repos you own.",
        )

        st.divider()
        st.subheader("Repository")
        repo_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/owner/repo",
            label_visibility="collapsed",
        )

        if st.button("Index repository", type="primary", use_container_width=True):
            index_repository(repo_url)

        if st.session_state.repo_label:
            st.success(f"Indexed: `{st.session_state.repo_label}`")
            st.caption(
                f"{st.session_state.doc_count} files · "
                f"{st.session_state.index_seconds:.1f}s to embed"
            )

        st.divider()
        top_k = st.slider(
            "Chunks retrieved per question", 2, 15, 5,
            help="Higher = more context for the model, slower and more tokens.",
        )
        st.session_state.top_k = top_k

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Built with LlamaIndex + Nebius AI · [Source](https://github.com/tirth1263/Codebase-QnA-RAG)")


def index_repository(repo_url: str) -> None:
    nebius_key = get_secret(st.session_state.nebius_key, "NEBIUS_API_KEY")
    github_token = get_secret(st.session_state.github_key, "GITHUB_TOKEN")

    if not repo_url:
        st.warning("Paste a GitHub repository URL first.")
        return
    if not nebius_key:
        st.error("A Nebius API key is required.")
        return
    if not github_token:
        st.error("A GitHub token is required to read the repository.")
        return

    try:
        owner, repo, branch = parse_github_url(repo_url)
    except ValueError as exc:
        st.error(str(exc))
        return

    started = time.time()
    try:
        with st.spinner(f"Reading {owner}/{repo}@{branch}…"):
            docs = load_repository(github_token, owner, repo, branch)
        if not docs:
            st.error("No supported source files found in that repository or branch.")
            return
        with st.spinner(f"Embedding {len(docs)} files…"):
            index = build_index(docs, nebius_key, f"{owner}/{repo}@{branch}")
    except Exception as exc:  # surface loader/API failures in the UI rather than the console
        st.error(f"Failed to index repository: {exc}")
        return

    st.session_state.index = index
    st.session_state.repo_label = f"{owner}/{repo}@{branch}"
    st.session_state.doc_count = len(docs)
    st.session_state.index_seconds = time.time() - started
    st.session_state.messages = []
    st.rerun()


def answer(question: str) -> tuple[str, list]:
    """Run one retrieval + synthesis pass, streaming tokens into the chat bubble."""
    configure_models(get_secret(st.session_state.nebius_key, "NEBIUS_API_KEY"))

    query_engine = st.session_state.index.as_query_engine(
        similarity_top_k=st.session_state.top_k,
        streaming=True,
    )
    query_engine.update_prompts({"response_synthesizer:text_qa_template": QA_PROMPT})

    response = query_engine.query(question)
    text = st.write_stream(response.response_gen)
    return text, response.source_nodes


def render_sources(source_nodes) -> None:
    if not source_nodes:
        return
    files = []
    for node in source_nodes:
        path = node.metadata.get("file_path") or node.metadata.get("file_name", "unknown")
        if path not in files:
            files.append(path)
    with st.expander(f"Sources ({len(files)} files)"):
        for path in files:
            st.markdown(f"- `{path}`")


def main() -> None:
    st.set_page_config(page_title="Codebase Q&A RAG", page_icon="🧠", layout="wide")

    defaults = {
        "messages": [], "index": None, "repo_label": "", "doc_count": 0,
        "index_seconds": 0.0, "top_k": 5, "nebius_key": "", "github_key": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    st.title("🧠 Codebase Q&A RAG")
    st.caption(
        "Point it at any GitHub repository and ask questions in plain English. "
        "Retrieval-augmented generation over source code, powered by LlamaIndex, "
        "Nebius embeddings and DeepSeek-V3."
    )

    render_sidebar()

    if st.session_state.index is None:
        st.info(
            "**Get started** — add your Nebius API key and GitHub token in the sidebar, "
            "paste a repository URL, then hit **Index repository**."
        )
        st.markdown(
            "Once a repo is indexed, try questions like:\n"
            "- *What does this project do, and how is it structured?*\n"
            "- *Where is authentication handled?*\n"
            "- *How do I add a new endpoint?*\n"
            "- *Explain the data flow from request to database.*"
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    if question := st.chat_input(f"Ask about {st.session_state.repo_label}…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                text, sources = answer(question)
                render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": text, "sources": sources}
                )
                st.download_button(
                    "Download answer",
                    data=text,
                    file_name="answer.md",
                    mime="text/markdown",
                )
            except Exception as exc:
                st.error(f"Query failed: {exc}")


if __name__ == "__main__":
    main()
