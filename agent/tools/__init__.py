"""Phase tool modules. Each owns one phase concern.

Dependency rule: tools/ may import agent.memory, agent.errors, agent.config,
agent.run_log, agent.retry — but never agent.orchestrator (no circular imports).
"""
