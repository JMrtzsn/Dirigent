# Dirigent Roadmap

## Next Up

- [ ] **Negotiate pattern** — agents propose/counter-propose until convergence. Novel primitive that no other framework does well.
- [ ] **Second LLM provider** — Anthropic direct or OpenAI. Proves the provider protocol is a real abstraction.
- [ ] **Integration test** — run the example end-to-end against a live LLM, assert on trace structure.
- [ ] **PyPI publish** — `pip install dirigent`. Package is ready, just needs the release plumbing.
- [ ] **Pipeline pattern** — sequential agent chains (A→B→C) as a first-class pattern.

## Future

- [ ] Network-backed channel (WebSocket/gRPC) for multi-process agents
- [ ] Persistent checkpointing (SQLite) for long-running orchestrations
- [ ] Web UI — visual graph execution, approval buttons, diff viewer
- [ ] Plugin system for custom providers
- [ ] Token budget / context window management
