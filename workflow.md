# RepoGPT — Workflow and Deep-Dive Study Guide

> **Purpose:** A continuously maintained, source-grounded guide to understanding RepoGPT's architecture, workflows, implementation, and design trade-offs. This document is intended for developers studying or contributing to the project.
>
> **Source snapshot reviewed:** `main` at commit `990a78f2e99e861bf8ec160d4a4f368a37542d05` (reviewed 29 October 2026). The source code is authoritative; this guide describes the code at that snapshot and should be updated when implementation changes.

## Contents

- [1. Project in one minute](#1-project-in-one-minute)
- [2. Repository map](#2-repository-map)
- [3. Technology stack and responsibility boundaries](#3-technology-stack-and-responsibility-boundaries)
- [4. System architecture](#4-system-architecture)
- [5. Workflow A — application startup](#5-workflow-a--application-startup)
- [6. Workflow B — indexing a GitHub repository](#6-workflow-b--indexing-a-github-repository)
- [7. Workflow C — answering a question](#7-workflow-c--answering-a-question)
- [8. Function-by-function code map](#8-function-by-function-code-map)
- [9. State, caching, and lifecycle](#9-state-caching-and-lifecycle)
- [10. Configuration and dependencies](#10-configuration-and-dependencies)
- [11. Errors, security, and limitations](#11-errors-security-and-limitations)
- [12. Glossary](#12-glossary)
- [13. Learning and verification log](#13-learning-and-verification-log)
- [14. Topics to add as the deep dive continues](#14-topics-to-add-as-the-deep-dive-continues)
- [15. Lesson 2: Embeddings and semantic retrieval](#15-lesson-2-embeddings-and-semantic-retrieval)

## 1. Project in one minute

RepoGPT is a Streamlit application for asking natural-language questions about a GitHub repository. It fetches supported files, builds an in-memory vector index using LlamaIndex and local Hugging Face embeddings, retrieves relevant chunks for a question, and uses a Groq-hosted language model to generate an answer. The UI shows source file paths associated with retrieved nodes.

RepoGPT is a **retrieval-augmented generation (RAG) application**. It does not train a new language model. It combines repository ingestion, embedding-based retrieval, prompt construction, hosted LLM generation, and a web interface.

### High-level lifecycle

~~~mermaid
flowchart TD
    A[Open Streamlit app] --> B[Initialize session defaults]
    B --> C{A vector index is available?}
    C -- No --> D[Show setup instructions]
    C -- Yes --> E[Show chat history and chat input]
    D --> F[User enters credentials and repository URL]
    F --> G[Index repository]
    G --> H[Save index in session state]
    H --> I[Rerun Streamlit script]
    I --> E
    E --> J[User submits a question]
    J --> K[Retrieve relevant nodes and generate answer]
    K --> L[Render answer and source paths]
    L --> M[Save message in session state]
~~~

**Key distinction:** indexing prepares a searchable representation of the repository; question answering uses that representation. They are separate operations.

## 2. Repository map

The current default branch is compact:

| Path | Responsibility |
|---|---|
| `main.py` | Application startup, NLTK setup, configuration, URL parsing, ingestion, indexing, chat, retrieval, source rendering and UI |
| `requirements.txt` | Python dependency declarations |
| `.env.example` | Example environment-variable configuration; contains blank credential values |
| `.gitignore` | Excludes local environment files, virtual environments, Python bytecode and Streamlit secrets |
| `.streamlit/config.toml` | Streamlit theme and server/browser settings |
| `README.md` | User-facing explanation, setup guide, architecture, limitations and deployment link |

The complete current tree was inspected. A separate branch, `fix/gemini-local-embeddings`, also exists, but this guide initially focuses on the implementation in `main`.

### Recommended source-reading order

1. Read the module-level setup and constants near the top of `main.py`.
2. Read `parse_github_url`, then the model/reader/index helpers.
3. Follow `render_sidebar` into `index_repository`.
4. Follow `answer` into LlamaIndex's query engine.
5. Finish with `render_sources` and `main`.
6. Compare each behavior with `requirements.txt`, `.env.example`, and `README.md`.

## 3. Technology stack and responsibility boundaries

| Component | How RepoGPT uses it | What to learn |
|---|---|---|
| Python | Application logic and orchestration | Functions, type hints, exceptions, modules, environment variables |
| Streamlit | UI, widgets, chat, session state, caching and streaming display | Reruns, widgets, `st.session_state`, `st.cache_resource`, `st.write_stream` |
| LlamaIndex | GitHub reader integration, document indexing, vector index and query engine | Documents/nodes, embeddings, retrievers, query engines, response synthesis |
| Hugging Face embedding integration | Loads `BAAI/bge-small-en-v1.5` for local text embeddings | Vector representations, semantic similarity and retrieval quality |
| Groq integration | Calls a language model hosted by Groq to synthesize answers | API configuration, model identifiers, prompt context, temperature, streaming and limits |
| GitHub reader/client | Reads supported files from a repository branch using a GitHub token | Repository paths, branches, permissions, remote APIs and ingestion failures |
| python-dotenv | Loads local `.env` values into the process environment | Secret/config separation and environment precedence |
| nest-asyncio | Applies event-loop compatibility support for dependencies that use asynchronous operations | Python event loops and library compatibility |
| NLTK | Ensures the configured stopwords and punkt_tab resources are present | Runtime resources and writable cache/data directories |

The code delegates several complex operations to libraries. For example, RepoGPT calls LlamaIndex to construct and query a vector index; it does not implement the vector-search algorithm from scratch.

## 4. System architecture

### 4.1 Indexing pipeline

~~~mermaid
flowchart TD
    A[Repository URL] --> B[parse_github_url]
    B --> C[Owner, repository and branch]
    C --> D[load_repository]
    D --> E[GithubClient and GithubRepositoryReader]
    E --> F[Filter supported extensions]
    F --> G[Repository documents and metadata]
    G --> H[build_index]
    H --> I[Configure LLM and embedding model]
    I --> J[LlamaIndex VectorStoreIndex.from_documents]
    J --> K[Embedding and indexing pipeline]
    K --> L[In-memory vector index]
    L --> M[Save index and repository metadata in session state]
~~~

The reader is configured with an inclusion filter and a list of supported extensions. The list includes common programming languages, notebooks, configuration files and documentation, but not every possible repository file type.

The in-memory vector index is not configured with a persistent external vector database. In the current implementation, restarting the process loses the index and it must be rebuilt.

### 4.2 Question-answering pipeline

~~~mermaid
flowchart TD
    A[Chat input] --> B[answer question]
    B --> C[Read Groq key and configure models]
    C --> D[Create query engine from session index]
    D --> E[Set similarity_top_k and streaming]
    E --> F[Apply QA_PROMPT]
    F --> G[query_engine.query]
    G --> H[Retrieve relevant source nodes]
    H --> I[Build answer from retrieved context]
    I --> J[Stream response.response_gen]
    J --> K[Display answer]
    H --> L[Return source_nodes]
    K --> M[Render unique source file paths]
    L --> M
    M --> N[Save assistant message and sources]
    N --> O[Offer Markdown download]
~~~

Retrieval and generation are distinct:
- **Retrieval** selects source nodes relevant to the question.
- **Generation/synthesis** uses the question and retrieved context to produce an explanation.
- **Source display** shows file paths from node metadata. These are useful navigation aids, but they are not necessarily exact line-number citations.

### 4.3 External services and local components

~~~mermaid
flowchart LR
    U[User browser] <--> S[Streamlit app]
    S --> GH[GitHub API / reader]
    GH --> S
    S --> EM[Local Hugging Face embedding model]
    EM --> VI[In-memory vector index]
    VI --> S
    S --> GQ[Groq-hosted LLM API]
    GQ --> S
~~~

The embedding model runs locally after its model files have been downloaded. Answer generation uses Groq, so retrieved repository context used to generate an answer is sent to that external provider. Review repository confidentiality and provider policies before indexing sensitive code.

## 5. Workflow A — application startup

### Module execution order

Python executes module-level statements before reaching the guarded entry point at the bottom of `main.py`.

1. Import standard library and third-party modules that appear before the NLTK setup.
2. Call `load_dotenv()`.
3. Resolve the NLTK data directory from `NLTK_DATA` or a path beside the source file; assign it to the process environment.
4. Import NLTK, create the data directory, and check for `stopwords` and `punkt_tab`. Missing resources are downloaded; a failed download raises an error.
5. Import LlamaIndex, Hugging Face and Groq integration classes.
6. Apply `nest_asyncio` and call `load_dotenv()` again.
7. Resolve the configured LLM/embedding model names and define the supported extensions and prompt template.
8. Define functions; these definitions do not themselves execute their function bodies.
9. The `if __name__ == "__main__"` guard calls `main()` when the script is run as the application module.

### Streamlit entry point

Inside `main()`, the app sets page metadata, initializes missing session-state keys with defaults, renders the title/caption and sidebar, then checks whether a vector index exists.

- If `st.session_state.index is None`, the app shows instructions and examples and returns from the current execution of `main()`.
- If an index exists, the app renders prior messages and the chat input.
- Streamlit may rerun the script after widget interaction. Session state preserves values across those reruns within the session.

### State lifecycle

~~~mermaid
stateDiagram-v2
    [*] --> NotIndexed
    NotIndexed --> Indexing: Click Index repository
    Indexing --> NotIndexed: Validation or indexing error
    Indexing --> Indexed: Index created successfully
    Indexed --> Indexed: Ask question / rerun
    Indexed --> NotIndexed: Process restart or session state reset
~~~

The final transition is a lifecycle concept, not a guarantee that every session reset behaves identically. The important implementation fact is that the index is held in runtime memory/session state rather than saved to a persistent database.

## 6. Workflow B — indexing a GitHub repository

The indexing button in `render_sidebar()` calls `index_repository(repo_url)`.

### Step-by-step

1. **Resolve credentials.** `get_secret` prefers a non-empty sidebar/session value; otherwise it uses the named environment variable; otherwise it returns an empty string.
2. **Validate inputs.** The function checks that a repository URL, Groq key and GitHub token are present. Missing inputs cause a UI message and an early return.
3. **Parse the URL.** `parse_github_url` validates the HTTP/HTTPS scheme and GitHub host, extracts owner and repository, strips a trailing `.git` suffix, and uses `main` unless a branch-specific `/tree/...` URL is supplied. Branch paths can contain multiple segments.
4. **Start timing.** The timer begins before the reader call and includes repository loading and index creation.
5. **Fetch documents.** `load_repository` creates a `GithubClient` and a `GithubRepositoryReader`, configures the inclusion filter and up to five concurrent requests, and calls `load_data(branch=branch)`.
6. **Handle an empty result.** If no supported documents are returned, the function shows an error and returns without setting the new index.
7. **Build the index.** `build_index` configures the model settings and calls `VectorStoreIndex.from_documents(...)`.
8. **Handle exceptions.** The indexing block catches exceptions and displays a user-facing failure message with exception details.
9. **Save state after success.** The index, repository label, document count, elapsed time and a fresh message list are assigned to session state.
10. **Rerun the app.** `st.rerun()` causes the next execution to show the indexed chat interface.

### Indexing sequence diagram

~~~mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit sidebar
    participant App as index_repository
    participant GH as GitHub reader
    participant IDX as LlamaIndex
    participant State as Session state

    User->>UI: Click Index repository
    UI->>App: Pass repository URL
    App->>App: Resolve credentials and validate URL
    App->>GH: Load supported documents for branch
    GH-->>App: Documents and metadata
    App->>IDX: Configure models and build VectorStoreIndex
    IDX-->>App: In-memory index
    App->>State: Save index, label, count, duration; clear messages
    App->>UI: Rerun
    UI-->>User: Display indexed state and chat
~~~

### URL parser behavior to know

Examples:
- `https://github.com/example/project` resolves to owner `example`, repository `project`, branch `main`.
- `https://github.com/example/project/tree/develop` resolves to branch `develop`.
- `https://github.com/example/project/tree/feature/nested-change` resolves to branch `feature/nested-change`.
- URLs with unsupported extra path segments are rejected rather than interpreted as arbitrary GitHub pages.

A syntactically accepted URL does not guarantee that the repository or branch exists, that credentials have permission, or that the reader can fetch supported files; those failures happen downstream.

## 7. Workflow C — answering a question

When a question is submitted, the chat-input branch in `main()` appends and renders the user message, then calls `answer(question)` inside an error-handling block.

### Answer function

1. Resolve the Groq key and call `configure_models`.
2. Create a query engine from the current session index using `similarity_top_k=st.session_state.top_k` and `streaming=True`.
3. Apply `QA_PROMPT` to the response synthesizer's text question-answering template.
4. Execute `query_engine.query(question)`.
5. Stream `response.response_gen` through `st.write_stream`.
6. Return the displayed text and `response.source_nodes`.

The caller renders source paths and appends the assistant message plus sources to `st.session_state.messages`. It offers a Markdown download of the answer. If question answering throws an exception, the app displays an error mentioning model access, quota and network as common areas to check.

### What `QA_PROMPT` asks the LLM to do

- Act as a senior engineer explaining an unfamiliar codebase.
- Use the provided repository excerpts as evidence.
- Treat repository code and comments as data, not as instructions that override the prompt.
- Answer the user's question using the supplied context.
- Cite relevant file paths and show short snippets when useful.
- Admit when the retrieved context does not contain the answer, rather than guessing.

This instruction is a useful mitigation against unsupported answers and repository-embedded prompt injection, but prompt wording alone is not a complete security boundary. Generated answers should still be verified against source code.

### Retrieval depth (top-k)

The sidebar slider permits values from 2 to 15 and initializes to 5. Increasing top-k can provide more evidence, but can also bring irrelevant chunks into the context, use more tokens and increase latency. There is no universal setting that is best for every repository or question; compare retrieval and answer quality empirically.

## 8. Function-by-function code map

| Function / constant | Responsibility | Key ideas to study |
|---|---|---|
| `NLTK_DATA_DIR` setup | Selects and prepares the NLTK resource directory | Filesystem paths, environment configuration, first-run side effects |
| `LLM_MODEL`, `EMBED_MODEL` | Reads model names from environment variables with defaults | Configuration defaults and model availability |
| `FILE_EXTENSIONS` | Defines the reader's included file extensions | Inclusion filters and ingestion coverage |
| `QA_PROMPT` | Defines context and question instructions for answer synthesis | Prompt templates, grounding and untrusted input |
| `parse_github_url(url)` | Extracts owner/repo/branch or raises a `ValueError` | URL parsing, decoding and validation |
| `get_embedding_model()` | Loads a Hugging Face embedding model under a resource cache | Expensive-resource reuse and embeddings |
| `configure_models(groq_api_key)` | Configures the global LlamaIndex LLM and embedding settings | API configuration and global settings |
| `load_repository(...)` | Reads repository documents and caches the result | GitHub access, branch selection, reader configuration |
| `build_index(_docs, groq_api_key, cache_key)` | Configures models and builds the vector index | Index creation and cache keys |
| `get_secret(session_value, env_var)` | Chooses a session or environment credential | Precedence and secret handling |
| `render_sidebar()` | Renders inputs, repository action, top-k slider and Clear chat | Streamlit widgets and callbacks |
| `index_repository(repo_url)` | Coordinates validation, ingestion, index creation and state changes | Orchestration and error handling |
| `answer(question)` | Runs retrieval and streamed answer synthesis | Query engines, prompts and source nodes |
| `render_sources(source_nodes)` | Extracts and displays unique file paths from node metadata | Metadata access and source navigation |
| `main()` | Initializes the UI and manages the chat lifecycle | Session state, reruns and control flow |

## 9. State, caching, and lifecycle

### Session state versus resource cache

These solve different problems:

- **Session state** holds user/session data, including the selected index reference, repository label, message history and top-k setting.
- **Resource caching** lets Streamlit reuse expensive objects or results between executions when its cache-key rules say a cached result can be reused.

The three cached functions are `get_embedding_model`, `load_repository` and `build_index`.

### A cache detail that needs testing

`build_index` has the signature `build_index(_docs, groq_api_key, cache_key)`. Streamlit excludes underscore-prefixed parameters from the ordinary cache-key calculation. The explicit `cache_key` participates in the cache key, but the function body does not use it.

Consequences worth investigating:
- The same repository URL/branch can produce the same explicit key after the repository content has changed.
- The Groq API key is a cache-key parameter even though it is not itself an input to the vector representation; a key change can cause unnecessary index rebuilding.
- Do not assume that changing document contents will invalidate the cached index; test this behavior with the installed Streamlit version.

A robust future design should use a deliberate content/version identifier for cache invalidation and avoid coupling index caching unnecessarily to credentials used only for answer generation. Any change should be accompanied by regression tests.

### In-memory index

No external persistent vector database is configured in the current implementation. The index is rebuilt after process restarts. Persistence, incremental updates and sharing an index across independent sessions are potential architectural extensions, not current features.

## 10. Configuration and dependencies

### Environment variables

The example environment file includes:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Credential for generating answers through Groq |
| `GITHUB_TOKEN` | Credential used by the GitHub reader |
| `GROQ_MODEL` | Optional Groq model identifier; source default is `openai/gpt-oss-120b` |
| `EMBED_MODEL` | Optional local embedding model; source default is `BAAI/bge-small-en-v1.5` |

The README recommends Python 3.11 or 3.12. The requirements file pins Streamlit to `1.50.0`, declares the LlamaIndex integrations, and constrains NLTK to `>=3.9.4,<3.10.0`.

The model identifier configured in the environment must be available to the current Groq account. Model availability can change over time, so confirm it in Groq's current model documentation instead of assuming a model will always be available.

## 11. Errors, security, and limitations

### Error handling already implemented

| Situation | Current behavior |
|---|---|
| Empty repository URL | Shows a warning and returns |
| Missing Groq key | Shows an error and returns |
| Missing GitHub token | Shows an error and returns |
| Invalid URL | Shows the parser's validation message and returns |
| No supported documents | Shows an error and returns |
| Exception during repository reading/indexing | Shows a broad failure message plus exception details |
| Exception during answering | Shows a model/quota/network-oriented error plus exception details |

### Security and reliability considerations

1. **Secrets:** Never commit `.env`, share API keys, or paste credentials into public issues. Sidebar password inputs mask display but do not, on their own, establish every aspect of credential isolation or retention in a deployed environment.
2. **Third-party inference:** Retrieved repository content used for answer generation is sent to Groq. Do not index confidential code unless its disclosure is permitted and the provider/deployment setup meets the applicable requirements.
3. **Least privilege:** Use a GitHub token with only the access required for the repositories to be read.
4. **Untrusted source content:** Repository files can contain misleading content or prompt injection attempts. Retrieved code should be treated as untrusted data.
5. **Hallucinations and retrieval misses:** The system can fail to retrieve the correct code or the LLM can draw an incorrect conclusion. Verify important claims against the referenced source.
6. **Source precision:** The interface displays file paths from metadata, not guaranteed line-level citations.
7. **Resource cost:** Large repositories can take longer to fetch and embed and may use substantial memory.
8. **External dependency:** Answer generation depends on Groq network availability, model access and account limits.
9. **In-memory lifecycle:** A process restart loses the current in-memory index.
10. **Exception disclosure:** The interface includes exception details in user-facing messages. A production deployment should assess whether details need redaction or structured logging.

### Potential improvements (not implemented in the current baseline)

- Add persistent vector storage and an explicit re-index/refresh policy.
- Make cache invalidation depend on repository version or content rather than only a repository label.
- Add automated tests for URL parsing, ingestion, caching and query flow.
- Add more precise clickable file and line-level citations.
- Evaluate retrieval with a repeatable question-and-source test set.
- Add structured logs, better error categorization and sanitized user-facing messages.
- Consider reranking, language-aware/code-aware chunking and repository-size safeguards.
- Review concurrency implications of global LlamaIndex `Settings` in a multi-user deployment.

Treat these as possible improvements to investigate and implement—not as features that already exist.

## 12. Glossary

- **RAG:** Retrieval-Augmented Generation; retrieve relevant information and provide it to a language model for answer generation.
- **Document:** An ingested source item with text and metadata.
- **Node / chunk:** A smaller text unit produced from a document for indexing and retrieval.
- **Embedding:** A numeric vector representing text in a semantic space.
- **Vector index:** A structure that organizes embedded nodes for similarity-based retrieval.
- **Top-k:** The number of highest-ranked retrieval results requested.
- **Query engine:** LlamaIndex's interface for querying an index and synthesizing a response.
- **Prompt template:** A reusable prompt containing placeholders for retrieved context and a question.
- **Source node:** A retrieved node included in the response metadata.
- **Session state:** Streamlit state associated with a user's active session.
- **Resource cache:** Streamlit mechanism for reusing expensive resources/results across reruns.
- **Streaming:** Displaying generated output incrementally rather than waiting for all output to finish.

## 13. Learning and verification log

Update this table as each teaching session is completed. A topic should be marked complete only after its code path has been inspected and the relevant experiment has been performed.

| Topic | Status | Evidence / next action |
|---|---|---|
| Repository structure and README | Reviewed | Current `main` branch tree and README inspected |
| Application startup and function call graph | Explained | Trace the guarded entry point, module startup and `main()` |
| Indexing workflow | Explained | Validate URL → load docs → build index → save session state |
| Question-answering workflow | Explained | Query engine → retrieval/synthesis → streaming → sources |
| Session state and caching | Initial review | Test cache behavior when repository content changes without changing URL/branch |
| Embeddings and semantic similarity | Lesson started | Conceptual explanation added in Section 15; run and record the retrieval experiment before marking complete |
| LlamaIndex documents, nodes, index and query engine | Pending deep dive | Inspect nodes/metadata and change retrieval settings |
| Groq generation, tokens and streaming | Pending hands-on lesson | Compare a direct API call with the LlamaIndex path |
| URL parsing and GitHub ingestion tests | Pending hands-on lesson | Add tests for standard, branch-specific and invalid URLs |
| Security, reliability and deployment trade-offs | Initial checklist | Validate actual deployment configuration and test improvements |
| End-to-end reimplementation | Pending | Build a small local-file RAG version without copying the full app |

## 14. Topics to add as the deep dive continues

This file is designed to grow during the study process. Add a dated subsection to the relevant chapter when a concept has been investigated. Prefer evidence from the current source, small reproducible experiments and links to official documentation.

Recommended next chapters:

1. **Embeddings in RepoGPT:** model inputs/outputs, vector dimension, similarity, normalization, and actual retrieval experiments.
2. **LlamaIndex internals used here:** document creation, chunking defaults, node metadata, index construction, retrieval and synthesis.
3. **Cache investigation:** reproduce behavior with unchanged URL and changed repository contents; document results and any fix.
4. **GitHub ingestion tests:** URL edge cases, branch permissions, extension filtering, failures and metadata.
5. **LLM request and streaming lifecycle:** prompt assembly, context budget, response generator and error cases.
6. **Testing strategy:** unit tests, integration tests, retrieval evaluation and end-to-end acceptance criteria.
7. **Production architecture:** persistence, cache invalidation, resource constraints, access control and safe observability.
8. **Interview preparation:** a five-minute architecture narrative, design trade-offs and evidence-backed answers to likely questions.

---

## 15. Lesson 2: Embeddings and semantic retrieval

**Status:** Conceptual walkthrough added; hands-on retrieval experiment remains pending. Do not mark this lesson complete until retrieval results have been observed and recorded.

### 15.1 The problem embeddings solve

Keyword search looks for literal text matches. Semantic retrieval tries to find text that is related in meaning, even when the wording differs.

For example, a developer might ask, “Where does the app check a user's login details?” while the code uses names such as `authenticate_user`, `verify_credentials`, or `login_handler`. A semantic search system can sometimes connect those concepts even without exact word overlap. That is an advantage, not a guarantee: retrieval quality depends on the embedding model, the code chunks, the query wording and the index configuration.

### 15.2 What an embedding is

An embedding model maps text to a numeric vector. Text with related meanings can produce vectors that are closer under a chosen similarity measure.

Conceptually:

~~~mermaid
flowchart LR
    A[Source text chunk] --> B[Embedding model]
    B --> C[Numeric vector]
    D[Natural-language question] --> E[Same embedding model]
    E --> F[Query vector]
    C --> G[Similarity ranking]
    F --> G
    G --> H[Top-k chunks]
~~~

RepoGPT configures the local embedding model by default as `BAAI/bge-small-en-v1.5`. Its source sets `Settings.embed_model = get_embedding_model()` inside `configure_models()`. The model is loaded through LlamaIndex's `HuggingFaceEmbedding` integration and cached by Streamlit.

A vector is not a summary, a database record, a generated answer, or a manually assigned category. It is a numeric representation that allows text to be compared in vector space.

### 15.3 Where embeddings happen in this codebase

The application delegates embedding operations to LlamaIndex; there is no direct call to a low-level model `encode()` method in `main.py`.

**At indexing time:**
1. `load_repository()` returns LlamaIndex documents representing supported repository files and associated metadata.
2. `build_index()` calls `VectorStoreIndex.from_documents(_docs, show_progress=False)`.
3. LlamaIndex applies its document/node processing and embedding pipeline using the configured embedding model.
4. The resulting vectors and nodes form the in-memory index used for search.

**At question time:**
1. `answer(question)` creates a query engine from `st.session_state.index`.
2. The query engine processes the question and obtains a query embedding through the configured embedding integration as part of the retrieval pipeline.
3. The index ranks candidate nodes according to its retrieval implementation.
4. The query engine uses retrieved context to synthesize the answer with the configured LLM.

The chunking/splitting configuration is not explicitly set in `main.py`; the effective behavior depends on the LlamaIndex version and default transformations in the installed dependency set. Inspect actual nodes and package versions before asserting a precise chunk size, overlap, or default ranking configuration.

### 15.4 Similarity intuition

A common similarity measure for vectors is cosine similarity:

$$
\operatorname{cosine}(a,b)=\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}
$$

Here, the dot product is in the numerator, and the denominator scales it by the vector lengths. For non-zero vectors, cosine similarity compares the alignment of the vectors' directions. Do not assume RepoGPT exposes this score in its UI: the current UI displays source paths, and the exact scoring implementation is delegated to the indexing/retrieval library and configuration.

### 15.5 What top-k means in RepoGPT

The sidebar slider ranges from 2 to 15 and defaults to 5. Its value is assigned to `st.session_state.top_k` and passed as `similarity_top_k` to `index.as_query_engine(...)`.

- A small top-k can omit a second relevant implementation detail that is stored in another chunk.
- A larger top-k can provide broader context, but it can also add unrelated text, increase prompt size, raise latency and dilute the useful evidence.
- A larger value is not automatically better. The useful setting depends on the query and repository.

### 15.6 Embedding, retrieval and generation are different jobs

| Stage | Input | Output | Responsibility |
|---|---|---|---|
| Document embedding | Source text nodes | Vectors | Represent source text numerically |
| Query embedding | User's question | Query vector | Represent the information need |
| Retrieval | Query vector and indexed vectors/nodes | Ranked source nodes | Select potentially useful evidence |
| Answer synthesis | Question plus retrieved context | Natural-language answer | Explain the retrieved material |

The embedding model does not write the final answer. The Groq-hosted LLM does not need to create all the stored document embeddings on every question. Keeping those responsibilities separate is a core RAG concept.

### 15.7 Hands-on experiment to complete

Use the already deployed application; no installation steps are required.

1. Index `https://github.com/dikshaforsure/RepoGPT`.
2. Ask: “Which function creates the searchable index?”
3. Record the answer and all source paths shown by the app.
4. Ask the same question in different words: “Where are repository documents converted into a structure that can be searched?”
5. Compare whether the source paths overlap and whether the answers identify `build_index` and `VectorStoreIndex.from_documents`.
6. Set top-k to 2 and repeat; then set it to 10 and repeat. Record changes in sources, factual usefulness, irrelevant context and latency.
7. Inspect the referenced code in `main.py` and verify each answer independently.

Keep observations in a small table with columns: question wording, top-k, source paths returned, answer accuracy, missing evidence and notes. Do not invent numerical relevance scores: the app currently does not display them.

### 15.8 Questions you should be able to answer

- Why is semantic retrieval useful when developers ask questions in natural language?
- What does the embedding model return?
- At which points in RepoGPT are document and query embeddings used?
- Why does changing top-k potentially change answer quality?
- Does RepoGPT train its embedding model? **No; the current application loads a pre-trained model.**
- Does the current UI expose numeric similarity scores? **No; it displays source paths from retrieved node metadata.**
- Is the chunk size explicitly configured in `main.py`? **No; inspect the effective LlamaIndex defaults and actual nodes before stating exact chunking settings.**

### 15.9 Completion criteria

Mark the embeddings/retrieval lesson complete only after you can:
- explain document embeddings versus query embeddings;
- point to the index-building and query-engine lines in `main.py`;
- explain top-k trade-offs;
- run the experiment above and record what actually happened; and
- verify at least one generated explanation against the original source.

---

## Source references

- [Repository](https://github.com/dikshaforsure/RepoGPT)
- [Main application source](https://github.com/dikshaforsure/RepoGPT/blob/main/main.py)
- [Dependencies](https://github.com/dikshaforsure/RepoGPT/blob/main/requirements.txt)
- [README and setup instructions](https://github.com/dikshaforsure/RepoGPT/blob/main/README.md)
- [Environment template](https://github.com/dikshaforsure/RepoGPT/blob/main/.env.example)
- [Streamlit caching and state documentation](https://docs.streamlit.io/develop/api-reference/caching-and-state)
- [LlamaIndex documentation](https://docs.llamaindex.ai/)
- [BGE small English v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Groq API documentation](https://console.groq.com/docs)

**Maintenance rule:** Keep the guide aligned with the code. When behavior changes, update the diagrams, function map, limitations and learning log in the same change. Do not document a proposed improvement as implemented until code and tests confirm it.
