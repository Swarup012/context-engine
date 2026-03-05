![Tests](https://img.shields.io/badge/tests-106%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![MCP](https://img.shields.io/badge/MCP-compatible-purple)

# ContextEngine 🧠
> **Give any AI tool deep understanding of your codebase — automatically.**

When you ask AI about your code, most tools dump entire files into 
context — burning 15,000 to 50,000 tokens per question. That's slow, 
expensive, and the AI gets confused by irrelevant code.

ContextEngine uses dependency graphs to find **only the relevant 
functions** — typically 2,000 to 5,000 tokens instead. Same answer 
quality. 85–95% fewer tokens. Works natively inside **Cursor**, 
**Claude Desktop**, and **Claude Code** via MCP.

---

## The Problem

Every time you ask AI about your code, one of three things happens:

- **Wrong files get read:**  generic answer that misses your actual implementation
- **Entire codebase gets dumped:**  50,000 tokens, slow response, AI loses focus
- **You paste code manually:**  defeats the purpose of having an AI assistant

The root cause is that most tools treat code like plain text.
They don't understand that `validateToken()` calls `checkExpiry()` 
which depends on `config.tokenSettings`. They just grab whatever 
looks similar and hope for the best.

**ContextEngine understands your code's structure.** It builds a 
real dependency graph , so when you ask about authentication, it 
traces the entire call chain automatically and assembles exactly 
the right context. Nothing irrelevant. Nothing missing.

---

## See It In Action
```bash
$ context-engine assemble "why does login fail after token expires?"

Query Type:   causal
Focal Points: auth.validate_token, session.check_expiry

┌───────┬───────────┬────────┬────────────┐
│ Tier  │ Functions │ Tokens │ % of Total │
├───────┼───────────┼────────┼────────────┤
│ HOT   │         3 │  1,240 │      52.1% │
│ WARM  │         2 │    580 │      24.4% │
│ COLD  │         4 │    560 │      23.5% │
│ TOTAL │         9 │  2,380 │       1.6% │
└───────┴───────────┴────────┴────────────┘

Token Budget: 150,000 | Used: 2,380 | Remaining: 147,620

🔥 HOT — auth.validate_token     (full source, 480 tokens)
🔥 HOT — session.check_expiry    (full source, 390 tokens)
🔥 HOT — auth.refresh_token      (full source, 370 tokens)
🌡 WARM — middleware.auth_guard  (compressed, 310 tokens)
🌡 WARM — config.token_settings  (compressed, 270 tokens)
🧊 COLD — user.get_profile       (signature, score: 0.41)
🧊 COLD — db.find_session        (signature, score: 0.38)
```

**2,380 tokens** instead of 40,000+ tokens of whole files.
The AI gets exactly what it needs. Nothing more.

---

## How It's Different

| | Cursor Built-in | GitHub Copilot | ContextEngine |
|---|---|---|---|
| Context selection | Open files | Current file | Dependency graph |
| Tokens used | 15,000–50,000 | 5,000–10,000 | **2,000–5,000** |
| Understands call chains | ❌ | ❌ | ✅ |
| Multi-system questions | ❌ | ❌ | ✅ |
| Works on large codebases | Struggles | Struggles | ✅ |
| Auto-detects relevant files | Partial | ❌ | ✅ |

---

## Quick Start

**Requirements:** Python 3.12+, uv

**1. Install uv** (if you don't have it):
```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Install ContextEngine:**
```bash
git clone https://github.com/yourusername/context-engine
cd context-engine
uv tool install .
```

Two commands are now available globally:
- `context-engine` — CLI tool
- `context-engine-mcp` — MCP server for Cursor/Claude Desktop

**3. Set your API key** (Optional if you use MCP not CLI):
> **Needed for AI powered compression or CLI usage, not for Cursor/Claude usage**
```bash
export ANTHROPIC_API_KEY=your_key_here
export MODEL_PROVIDER=claude
```

That's it. You're ready.

---

## Using With Cursor (Recommended)

### Step 1 — Add to Cursor

Edit `~/.cursor/mcp.json` (create if it doesn't exist):
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

> If `context-engine-mcp` isn't found, use the full path:
> run `which context-engine-mcp` and paste that path instead.

### Step 2 — Restart Cursor

Go to **Settings → Features → MCP Servers** and verify
`context-engine` shows a green dot ✅.

### Step 3 — Index and Ask

Open your project in Cursor and say:
```
index this codebase
```

Then ask anything:
```
how does authentication work in this project?
why does the dashboard show stale data after login?
what is the difference between login and register?
what calls the validateToken function?
```

Cursor calls ContextEngine automatically. You just get answers.

---

## Using With Claude Desktop

**Find your config file:**

| Platform | Location |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\\Claude\\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

**Add ContextEngine:**
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

Restart Claude Desktop. The tools appear automatically.

---

## Using With Claude Code
```bash
# Add to ~/.claude.json
{
  "mcpServers": {
    "context-engine": {
      "command": "context-engine-mcp",
      "args": []
    }
  }
}
```

Then from inside your project:
```bash
cd /your/project
claude
# say: "index this codebase" and start asking questions
```

---

## MCP Tools Available

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

## CLI Usage

Use ContextEngine as a standalone tool without any IDE:
```bash
# Index your project (run once)
context-engine index /path/to/project

# Keep index fresh as you code
context-engine watch /path/to/project

# Ask a question (requires API key)
context-engine ask "how does payment processing work?" --path .

# See HOT/WARM/COLD breakdown without calling LLM
context-engine assemble "how does auth work?" --path .

# Semantic search
context-engine query "token validation" --path .

# Check index stats
context-engine status --path .
```

---

### Supported Languages

| Language | Extensions |
|---|---|
| Python | `.py` |
| JavaScript | `.js`, `.jsx` |
| TypeScript | `.ts`, `.tsx` |

---


## When To Re-Index

The index doesn't auto-update embeddings (by design — embedding
models are slow to reload). Re-index after significant code changes:
```bash
context-engine index .
# or in Cursor/Claude: "index this codebase"
```

The file watcher updates the dependency graph in real time:
```bash
context-engine watch .
```

Check if your index is stale:
```bash
context-engine status .
# or in Cursor/Claude: "is the codebase index up to date?"
```

---

## Configuration
```bash
# .env or system environment variables

MODEL_PROVIDER=claude          # claude | openai | gemini
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

Default token budget is **150,000 tokens** per query.
Override per query in CLI:
```bash
context-engine ask "question" --token-budget 50000 --path .
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
