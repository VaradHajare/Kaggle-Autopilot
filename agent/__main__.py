"""Module entrypoint so the agent runs as ``python -m agent ...``.

CLAUDE.md documents ``python -m agent run <url>`` as the primary invocation;
this delegates to the Typer app defined in ``agent.cli``.
"""

from __future__ import annotations

from agent.cli import app

if __name__ == "__main__":
    app()
