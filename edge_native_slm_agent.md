# Edge-native SLM parts triage agent

## How to use this document

This spec is written for Claude Code, with integrated learning checkpoints. After completing each phase, Claude Code will **stop and quiz you** before proceeding. Answer in your own words — the goal is to make sure the concepts are yours, not just the code.

Quizzes mix two types of questions:
- **Conceptual**: why does this work the way it does?
- **Interview-ready**: how would you explain this to an InstaLILY engineer?

You don't need a perfect answer to move on — but you should be able to say something genuine. Claude Code will fill gaps and correct misconceptions before continuing.

---

## Project overview

Build a fully local, edge-native agentic system that triages industrial parts and inventory requests using a small language model running entirely on-device. No cloud inference, no API keys, no outbound calls during operation.

The agent takes a natural language request ("do we have SKU 4821-B in stock, and if not who's the nearest supplier?"), reasons over a local knowledge base of parts and inventory, chains tool calls, and returns a structured triage response — all without touching the internet.

This project is scoped around the architectural pattern used by InstaLILY's InstaBrain™ and InstaWorkers™: domain-specific SLMs reasoning over editable enterprise knowledge, executing multi-step workflows inside operational systems.

---

## Why local/edge-native

This is not a stylistic choice — it is the core engineering constraint the project is built around. Understanding the distinction from cloud-based inference is important before writing any code.

**Cloud inference** (OpenAI, Anthropic, Google APIs): your application sends requests over the internet to a remote GPU cluster. The model never touches your machine. You are renting inference. Every LLM call is an outbound HTTP request with network latency, data leaving your environment, and a dependency on external uptime.

**Local edge-native inference** (this project): model weights live on your machine. Inference runs on your hardware — in this case, the M-series Neural Engine via Metal backend through ollama. No network hop. No data leaving the device. The LLM call is a local subprocess call.

The practical implications for this project:

- **Data sovereignty**: the parts catalog, pricing, and supplier data never leave the machine. This is a hard requirement in enterprise manufacturing and distribution contexts.
- **Connectivity independence**: the agent works on a factory floor or warehouse with no internet. Cloud-dependent agents fail here.
- **Latency**: no round-trip overhead. First-token latency is purely a function of local hardware and quantization level — typically faster than a cloud API for short prompts on M-series.
- **Cost at scale**: no per-token billing. Inference cost is amortized into the hardware, which matters for agents running continuously inside enterprise workflows.

The model must be quantized to run on consumer hardware. A full-precision 3B model is ~6GB. Q4 quantization brings it to ~2GB, making it viable in unified memory on a MacBook Air. Quantization is not optional — it is what makes edge-native inference possible.

---

## Target hardware

Primary: MacBook Air M-series (unified memory, Metal backend via ollama)

The project should run fully on this machine with no GPU required. All tooling choices below reflect this constraint.

---

## Tech stack

| Layer | Tool | Rationale |
|---|---|---|
| LLM inference runtime | `ollama` | Native Metal support on M-series, easy model management, local subprocess interface |
| Primary model | `Qwen2.5-3B-Instruct` (GGUF Q4_K_M) | Strong instruction following at 3B scale, fits comfortably in unified memory when quantized |
| Comparison model | `Phi-4-mini` | Benchmark against Qwen2.5-3B for model selection justification |
| Vector store | `ChromaDB` | Runs as a local process, no server required, good Python SDK |
| Agent framework | `LangGraph` | Explicit graph-based agent loop, good tool-calling support, works with local models via ollama |
| Embeddings | `nomic-embed-text` via ollama | Local embedding model, no API call required |
| Interface | CLI (Python) | Keeps scope tight for v1; demo-able via screen recording |
| Notebook | Jupyter | Benchmark documentation artifact |

**Do not introduce any dependency that requires an outbound network call during agent operation.** The only permitted network calls are during setup (pulling models via ollama, installing packages).

---

## Data

### Synthetic parts catalog (generate before coding)

Use GPT-4o or Gemini to generate a CSV with 200–300 rows of realistic industrial parts data. This mirrors InstaLILY's approach of using a teacher model to seed domain knowledge.

**Schema:**

| Field | Type | Example |
|---|---|---|
| `sku` | string | `HYD-4821-B` |
| `description` | string | `Hydraulic fitting, 3/8" NPT male, stainless` |
| `category` | string | `Hydraulic fittings` |
| `stock_qty` | integer | `47` |
| `unit_price` | float | `12.40` |
| `supplier` | string | `Parker Hannifin` |
| `supplier_lead_days` | integer | `5` |
| `reorder_threshold` | integer | `10` |

**Suggested prompt for generation:**
```
Generate a CSV of 250 industrial parts for a mid-size hydraulic and pneumatic components distributor.
Include a mix of in-stock and out-of-stock items (stock_qty = 0 for ~20% of rows).
Categories should include: hydraulic fittings, pneumatic valves, electrical connectors,
seals and gaskets, pressure gauges. Use realistic SKU formats, supplier names, and pricing.
Output only the CSV, no explanation.
```

Save the output as `data/parts_catalog.csv`.

---

## Project structure

```
edge-native-parts-agent/
├── README.md
├── data/
│   └── parts_catalog.csv
├── scripts/
│   └── ingest.py          # embeds catalog into ChromaDB
├── agent/
│   ├── __init__.py
│   ├── tools.py           # lookup_part, check_stock tool definitions
│   ├── graph.py           # LangGraph agent graph definition
│   └── memory.py          # ChromaDB read/write interface
├── cli.py                 # entry point, natural language → triage response
├── benchmarks/
│   └── inference_benchmark.ipynb
└── requirements.txt
```

---

## Phase 1 — Environment and data foundation

### Learning objectives
By the end of this phase you should be able to explain:
- What quantization does to a model and why it's necessary for local inference
- What an embedding is and what role it plays in making the knowledge base searchable
- Why the embedding model also needs to be local (not an API call)

### Tasks

**1. Install and verify ollama**
```bash
# Install ollama (macOS)
brew install ollama

# Pull the primary model (quantized)
ollama pull qwen2.5:3b

# Pull the local embedding model
ollama pull nomic-embed-text

# Verify
ollama run qwen2.5:3b "What is a hydraulic fitting?"
```

Confirm: response generates locally, no outbound API call, Metal backend active (check Activity Monitor for Neural Engine usage).

**2. Generate and save the synthetic catalog**

Use the prompt above with GPT-4o or Gemini (one-time use, during setup only). Save to `data/parts_catalog.csv`. Manually review a sample — check that ~20% of rows have `stock_qty = 0` and that SKUs follow a consistent format.

**3. Build the ingestion script (`scripts/ingest.py`)**

This script reads the CSV, generates embeddings for each part using the local embedding model, and stores them in ChromaDB.

Requirements:
- Use `ollama` Python SDK to call `nomic-embed-text` for embeddings (local, no API)
- ChromaDB collection named `parts_catalog`
- Store full row metadata alongside each embedding so retrieval returns all fields
- Script should be idempotent: re-running clears and re-ingests cleanly
- Print a confirmation with row count on completion

**4. Verify retrieval**

After ingestion, run a quick manual test: query the ChromaDB collection with a natural language string ("3/8 inch hydraulic fitting") and confirm the top results are semantically sensible. This validates the embedding pipeline before building the agent on top of it.

### Deliverable

Running `python scripts/ingest.py` successfully embeds the catalog. A manual similarity query returns relevant parts. The model responds to a test prompt via ollama CLI.

---

### 🎯 Checkpoint 1 — quiz before proceeding to Phase 2

> Claude Code: before writing any agent code, ask the user the following questions one at a time. Wait for a response to each before asking the next. Correct misunderstandings and fill gaps before moving on. Do not proceed to Phase 2 until all three are answered with reasonable understanding.

**Q1 (conceptual):** When you ran `ollama pull qwen2.5:3b`, you got a quantized model by default. In plain terms, what did quantization actually do to the model — and what did you trade off to get it to fit on your machine?

**Q2 (conceptual):** You stored the parts catalog in ChromaDB as embeddings rather than just loading the CSV directly. Why? What does an embedding give you that a keyword search over the raw CSV wouldn't?

**Q3 (interview-ready):** Imagine an InstaLILY engineer asks: "Why did you use a local embedding model instead of OpenAI's embedding API?" Give the answer you'd give in that conversation.

---

## Phase 2 — Core agent loop

### Learning objectives
By the end of this phase you should be able to explain:
- What a tool-calling agent is and how it differs from a simple LLM prompt
- What LangGraph's state graph gives you that a plain while loop wouldn't
- How the agent decides which tool to call and when to stop

### Tasks

**1. Define tools (`agent/tools.py`)**

Two tools for v1:

`lookup_part(query: str) -> list[dict]`
- Runs a similarity search against ChromaDB using the local embedding model
- Returns top 3 matching parts with all metadata fields
- Used when the request is descriptive ("do we have any 3/8 inch hydraulic fittings?")

`check_stock(sku: str) -> dict`
- Exact lookup by SKU against ChromaDB metadata
- Returns the full part record including `stock_qty`, `supplier`, `supplier_lead_days`
- Used when the request references a specific SKU

Both tools must use only local resources — ChromaDB query + ollama embedding. No HTTP calls.

**2. Build the agent graph (`agent/graph.py`)**

Use LangGraph to define the agent as an explicit state graph:

- State: `{ messages, current_request, tool_results, final_response }`
- Nodes: `reason` (LLM decides next action), `call_tool` (executes selected tool), `respond` (formats final triage output)
- Edges: reason → call_tool → reason (loop until done) → respond
- The LLM backend is the local ollama model, not any cloud API

Triage logic the agent should handle:
- Part found + in stock → return part details, stock level, unit price
- Part found + out of stock → return part details, supplier name, lead time
- Part not found → say so clearly, suggest searching by category
- Ambiguous request → agent asks a clarifying question before tool call

**3. Build the knowledge base interface (`agent/memory.py`)**

Wrapper around ChromaDB with methods:
- `search(query: str, n: int) -> list[dict]` — similarity search
- `get_by_sku(sku: str) -> dict` — exact SKU lookup
- `add_part(part: dict)` — add a new part at runtime
- `update_part(sku: str, fields: dict)` — update fields on an existing part
- `remove_part(sku: str)` — delete a part by SKU

The add/update/remove methods are used in Phase 4 for runtime editing. Stub them in Phase 2 so the interface is complete.

**4. CLI entry point (`cli.py`)**

Simple interactive loop:
```
$ python cli.py
Parts triage agent (local) — type a request or 'quit' to exit

> do we have any 3/8 inch hydraulic fittings in stock?
[agent reasoning...]
Found 3 matching parts. 2 are in stock:
  - HYD-4821-B: Hydraulic fitting 3/8" NPT male, stainless | Qty: 47 | $12.40/unit
  - HYD-4822-A: Hydraulic fitting 3/8" NPT female, brass | Qty: 12 | $9.80/unit
1 is out of stock:
  - HYD-4819-C: Hydraulic fitting 3/8" JIC male, stainless | Out of stock | Parker Hannifin | Lead: 5 days

> quit
```

Show the agent's tool calls as it reasons (useful for the demo video). Use a simple flag to toggle verbose reasoning output.

### Deliverable

`python cli.py` accepts natural language requests and returns grounded triage responses. The agent chains tool calls correctly for both in-stock and out-of-stock paths. No internet connection required.

---

### 🎯 Checkpoint 2 — quiz before proceeding to Phase 3

> Claude Code: the agent is running. Before benchmarking, ask the user the following questions one at a time. Wait for a response to each. Correct gaps before moving on.

**Q1 (conceptual):** Walk me through what happens — step by step inside the LangGraph graph — when you type "do we have SKU 4821-B in stock?" into the CLI. What does each node do, and how does the agent decide it's done?

**Q2 (conceptual):** You have two tools: `lookup_part` and `check_stock`. Why two? What would break or get worse if you collapsed them into a single tool that did both?

**Q3 (interview-ready):** An InstaLILY engineer asks: "How does your agent know when to stop calling tools and return an answer?" What's your answer?

**Q4 (interview-ready):** "Why LangGraph instead of just writing a while loop yourself?" Give the honest engineering answer.

---

## Phase 3 — Inference optimization and benchmarking

### Learning objectives
By the end of this phase you should be able to explain:
- The practical difference between Q4 and Q8 quantization in terms of tradeoffs
- What metrics actually matter for evaluating a model in an agentic context
- How to make and defend a model selection decision with empirical evidence

### Tasks

**1. Set up benchmark harness**

In `benchmarks/inference_benchmark.ipynb`, build a simple harness that:
- Sends a fixed set of 10 triage requests to the agent
- Records: time to first token, total generation time, peak memory usage, response quality score (manual 1-3 rating)
- Runs the same set against each config being tested

**2. Benchmark quantization levels on Qwen2.5-3B**

Configs to test:
- `qwen2.5:3b` (Q4_K_M — default ollama pull)
- `qwen2.5:3b-instruct-q8_0` (Q8)
- Full precision if memory allows

For each config, record the metrics above. Build a summary table.

**3. Benchmark Phi-4-mini**

```bash
ollama pull phi4-mini
```

Run the same 10 requests. Add to the summary table.

**4. Document findings**

The notebook should include:
- Summary table: model × quantization → latency / memory / quality
- A written conclusion: which config you chose and why, with explicit tradeoff reasoning
- Notes on what degraded first as you quantized (instruction following? factual recall? structured output?)

This notebook is a talking point in an interview. Write it as if someone who didn't build it will read it.

### Deliverable

A completed benchmark notebook with a clear final recommendation. The agent is updated to use the chosen config.

---

### 🎯 Checkpoint 3 — quiz before proceeding to Phase 4

> Claude Code: benchmarks are done. Ask these questions one at a time before moving on. Push back on vague or surface-level answers — this section is the one most likely to come up in an InstaLILY technical conversation.

**Q1 (conceptual):** What is the actual difference between Q4_K_M and Q8_0 at a mechanical level — what is being reduced, and what does "K_M" mean in the GGUF naming convention?

**Q2 (conceptual):** Looking at your benchmark results: what degraded first as you moved from Q8 to Q4? Was it latency, memory, or output quality — and did that match what you expected going in?

**Q3 (interview-ready):** "You benchmarked two models. How did you decide which one to use in the final agent?" Give the answer with the actual numbers from your notebook. Don't generalize — be specific about what you measured and why it drove your decision.

**Q4 (interview-ready):** "What metrics would you add if this agent were going into production in a warehouse environment?" Think beyond what you measured.

---

## Phase 4 — Editable memory and polish

### Learning objectives
By the end of this phase you should be able to explain:
- Why runtime editability matters in enterprise deployments (the business case)
- What makes a project README function as a portfolio artifact vs. just documentation
- How to frame this project in terms of InstaLILY's actual product architecture

### Tasks

**1. Activate runtime editing in the CLI**

Add CLI commands alongside the triage interface:

```
> /add  — prompts for part fields, adds to ChromaDB
> /update HYD-4821-B stock_qty 0  — updates a specific field
> /remove HYD-4821-B  — removes a part
> /list hydraulic fittings  — lists parts in a category
```

These use the `memory.py` methods stubbed in Phase 2. Demonstrate that a part added at runtime is immediately retrievable by the agent in the next query.

**2. Write the README**

Structure:
- What this is (one paragraph, plain language)
- Why edge-native (the business case — data sovereignty, connectivity, cost)
- Architecture diagram (ASCII or linked image)
- Stack with rationale for each choice
- How to run it (setup → ingest → agent)
- Benchmark results summary (link to notebook)
- What's next (vision layer stretch goal)

Write it for a technical reader who has not seen the project before. Frame the use case around industrial distribution — the physical economy context — not a generic "parts lookup tool."

**3. Record a 2-minute demo video**

Script:
1. Show the terminal — no browser, no API keys, no cloud dashboard open
2. Run `python cli.py`
3. Ask two requests: one in-stock, one out-of-stock, one with an ambiguous description
4. Show the agent reasoning through tool calls (verbose mode on)
5. Run a `/add` command, then immediately query for the new part to show live editability
6. End with a note on what the benchmark showed about quantization tradeoff

No narration required if the verbose output speaks for itself. Keep it under 2 minutes.

### Deliverable

Public GitHub repo with clean commit history, complete README, and a linked demo video. This is the artifact you send to InstaLILY.

---

### 🎯 Checkpoint 4 — final debrief before stretch goal

> Claude Code: the project is shippable. Run through these final questions. These are the ones most likely to come up if InstaLILY asks you to walk through the project in a conversation or screen share.

**Q1 (interview-ready):** "Walk me through your architecture in under 2 minutes." Do it out loud or in writing — no looking at the README. This is the screen-share moment.

**Q2 (interview-ready):** "How does this relate to what InstaLILY actually builds?" Connect the dots explicitly: InstaBrain™, InstaWorkers™, the physical economy thesis, why edge-native matters to their customers.

**Q3 (conceptual):** If you had to add a second domain — say, a separate catalog for electrical components from a different supplier — what would you change in the architecture? Where does it get complicated?

**Q4 (stretch thinking):** What's the weakest part of this system right now? If InstaLILY asked "what would you improve first?", what's your honest answer?

---

## Stretch goal — vision layer

> Revisit after all four phases are complete, all checkpoints passed, and the repo is public.

Add `moondream2` or `Qwen2.5-VL-3B` as a local vision model. The agent gains a `read_part_label(image_path: str) -> str` tool that:
- Accepts a photo of a physical part label or shelf tag
- Runs it through the local VLM to extract SKU or description text
- Passes the extracted text into the existing triage pipeline

This completes the "see, reason, act" loop from the InstaLILY hackathon spec. A warehouse worker photographs a part → agent reads the label → triages against the catalog → returns a decision. Entirely on-device.

Do not attempt this until phases 1–4 are working and benchmarked. The vision layer adds model size, memory pressure, and a new inference runtime to manage.

---

## Definition of done

- [ ] ollama running locally with chosen model, no outbound inference calls
- [ ] Parts catalog embedded in local ChromaDB instance
- [ ] Agent handles in-stock, out-of-stock, and ambiguous request paths correctly
- [ ] Runtime add/update/remove works without agent restart
- [ ] Benchmark notebook complete with model × quantization comparison table and written conclusion
- [ ] All four checkpoints passed with genuine understanding (not just code running)
- [ ] README written for an external technical reader
- [ ] Demo video recorded and linked from README
- [ ] GitHub repo public with clean history
