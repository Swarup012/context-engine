# ContextEngine 🧠

> **Give any AI tool deep understanding of your codebase — automatically.**

ContextEngine sits between your code and any LLM. Instead of you manually copying files or guessing what context to provide, ContextEngine automatically figures out which functions, files, and relationships are relevant to your question — and assembles the perfect context window.

Works natively inside **Cursor**, **Claude Desktop**, **Claude Code**, and any MCP-compatible tool.

---

## What Problem Does This Solve?

When you ask an AI "how does authentication work in my app?", you usually have to:
- Manually open the right files
- Copy-paste the relevant code
- Hope you didn't miss anything

**ContextEngine does all of that automatically.** It indexes your codebase once, builds a dependency graph, and then for any question — finds exactly the right code, organized by relevance.

```
You ask: "why does login fail after token expires?"
           ↓
ContextEngine finds:
  🔥 HOT  → validate_token() — full source (directly relevant)
  🌡 WARM → auth_middleware() — compressed summary (related)
  🧊 COLD → UserModel.save() — just the signature (peripherally relevant)
           ↓
AI gets perfect context. You get a precise answer.
```

---

## Quick Start

### Install

```bash
# Requires Python 3.12+ and uv
pip install uv  # if you don't have uv yet

# Clone and install
git clone https://github.com/yourusername/context-engine
cd context-engine
uv tool install .
```

This installs two global commands:
- `context-engine` — CLI tool
- `context-engine-mcp` — MCP server for AI tools

### Set Up API Keys (Optional)

Only needed if you want AI-powered WARM tier compression in the CLI. Not required for Cursor/Claude Desktop usage.

```bash
cp .env.example .env
# Edit .env and add your key:
# MODEL_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-...
```

---

## Using ContextEngine in Cursor (Recommended)

This is the best way to use ContextEngine — it works invisibly inside your normal Cursor workflow.

### Step 1: Add to Cursor

Edit `~/.cursor/mcp.json` (create it if it doesn't exist):

```json
{
  "mcpServers": {
    "context-engine": {
      "command": "context-engine-mcp",
      "args": []
    }
  }
}
```

> **Linux/Mac:** If `context-engine-mcp` isn't found, use the full path:
> `which context-engine-mcp` → use that full path in the config.

### Step 2: Restart Cursor

After restarting, go to **Cursor Settings → Features → MCP Servers** and verify `context-engine` shows a green dot ✅.

### Step 3: Open Your Project and Index It

Open your project folder in Cursor, then in the chat just say:

```
index this codebase
```

Cursor will call the `index_codebase` tool automatically. You'll see a summary like:

```
Indexed 47 files, 312 functions, 891 edges.
Index saved to /your/project/.context-engine
```

### Step 4: Ask Questions Naturally

That's it! Now just ask questions about your code:

```
How does authentication work in this project?
What does the compress command do?
Find all API routes
Show me the source of validateToken
What calls the login function?
What does auth.ts do?
Is the codebase index up to date?
```

---

## Using ContextEngine in Claude Desktop

### Step 1: Find Your Config File

| Platform | Config Location |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

### Step 2: Add ContextEngine

```json
{
  "mcpServers": {
    "context-engine": {
      "command": "context-engine-mcp",
      "args": [],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Step 3: Restart Claude Desktop

The context-engine tools will appear in Claude's tool picker automatically.

---

## Using ContextEngine in Claude Code (CLI)

```bash
# Add to your Claude Code config
# ~/.claude.json — add the mcpServers key:
{
  "mcpServers": {
    "context-engine": {
      "command": "/full/path/to/context-engine-mcp",
      "args": []
    }
  }
}
```

Then open Claude Code from inside your project folder:
```bash
cd /your/project
claude
```

Say `index this codebase` and you're ready to go.

---

## All 7 MCP Tools

Once connected, these tools are available to any MCP-compatible AI:

| Tool | What It Does | Example Trigger |
|---|---|---|
| `index_codebase` | Index a project directory | *"index this codebase"* |
| `ask_codebase` | Get assembled context for a question | *"how does auth work?"* |
| `get_codebase_status` | Check if index is up to date | *"is the index stale?"* |
| `search_codebase` | Find functions/files by concept | *"find all CLI commands"* |
| `get_function_source` | Get full source of a function | *"show me validateToken source"* |
| `explain_file` | Overview of a file's structure | *"what does auth.ts do?"* |
| `find_dependents` | Who calls a given function | *"what calls login()?"* |

> **Note:** All tools automatically use the project directory you opened your AI tool from. You rarely need to specify a path manually.

---

## Using the CLI Directly

You can also use ContextEngine as a standalone CLI without any AI tool:

```bash
# Index your project
context-engine index /path/to/your/project

# Search for relevant functions
context-engine query "token validation" --path /path/to/your/project

# Assemble context for a query (see HOT/WARM/COLD breakdown)
context-engine assemble "how does login work?" --path /path/to/your/project

# Ask a question (requires API key in .env)
context-engine ask "why does payment fail after timeout?" --path /path/to/your/project

# Check index status
context-engine status --path /path/to/your/project

# Watch for file changes and auto-update index
context-engine watch /path/to/your/project
```

---

## How It Works

### The Three Tiers

ContextEngine organizes code into three tiers based on relevance to your query:

```
🔥 HOT  — Full source code
         The function(s) most directly relevant to your question.
         You get every line, every comment.

🌡 WARM  — Compressed summaries
         Functions that are related (2 hops away in the dependency graph).
         Summarized to 2-3 sentences by an LLM to save tokens.

🧊 COLD  — Signatures only
         Functions at the periphery — just enough to know they exist
         and what they do, without filling your context window.
```

### The Pipeline

```
1. Index     → tree-sitter parses Python/JS/TS files
               → builds NetworkX dependency graph
               → generates ChromaDB semantic embeddings

2. Query     → LLM analyzes your question
               → detects type: single, multi, causal, comparison, or enumeration
               → finds 1-5 focal points via semantic search

3. Assemble  → graph traversal from each focal point
               → fills token budget: HOT → WARM → COLD
               → irrelevant COLD functions filtered out (threshold ≥ 0.3)

4. Return    → formatted context string to your AI tool
               → AI reasons over real code, not guesses
```

### Supported Languages

| Language | Extensions |
|---|---|
| Python | `.py` |
| JavaScript | `.js`, `.jsx` |
| TypeScript | `.ts`, `.tsx` |

---

## The .context-engine Folder

When you index a project, ContextEngine creates a `.context-engine/` folder inside it:

```
your-project/
├── .context-engine/
│   ├── graph.pkl          ← dependency graph
│   ├── functions.pkl      ← parsed function data
│   ├── metadata.json      ← stats + last indexed time
│   └── chroma_db/         ← semantic embeddings
├── src/
└── ...
```

This folder is automatically added to `.gitignore`. It's safe to delete — just re-run `index_codebase` to rebuild.

---

## Re-indexing

The index doesn't update automatically when you edit files (by design — embedding model loading is slow). Re-index whenever you make significant changes:

```
"index this codebase"   ← in Cursor/Claude chat
```
or
```bash
context-engine index .
```

To check if your index is stale:
```
"is the codebase index up to date?"
```

---

## Configuration

### .env File

```bash
# Which LLM to use for WARM tier compression and CLI ask command
MODEL_PROVIDER=anthropic        # anthropic | openai | gemini

# API Keys (only need the one you use)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# Optional: override default models
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
OPENAI_MODEL=gpt-4o-mini
GEMINI_MODEL=gemini-2.0-flash
```

### Token Budget

Default is 150,000 tokens per query — well within Claude's 200k context window. You can override per query:

```
"ask_codebase with token_budget 50000: how does auth work?"
```

---

## Requirements

- **Python 3.12+**
- **uv** package manager
- One of: Anthropic API key, OpenAI API key, or Gemini API key (for WARM compression + CLI ask)

---

## Troubleshooting

**MCP server not showing up in Cursor**
- Make sure you restarted Cursor fully after editing `mcp.json`
- Try using the full absolute path to `context-engine-mcp` in the config
- Check `Cursor Settings → Features → MCP Servers` for error messages

**"No index found" error**
- Run `index_codebase` first before asking questions
- Make sure the `path` points to the correct project directory

**Index is stale**
- Re-run `index_codebase` after making code changes
- The `get_codebase_status` tool will tell you which files changed

**AI not calling tools automatically**
- Try being explicit: *"use context-engine to answer: how does auth work?"*
- Larger models (Claude 3.5, GPT-4o) trigger tools more reliably than small models

**`context-engine-mcp` not found**
- Find the full path: `which context-engine-mcp`
- Use that full path in your MCP config instead of just `context-engine-mcp`

---

## License

MIT
