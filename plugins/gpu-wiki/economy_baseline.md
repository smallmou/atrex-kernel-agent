Create `wiki_request.json` with this bounded, framework-free request, preserving the exact
operator identifier and runtime architecture:

```json
{"request":"Target hardware {{PLATFORM}}, runtime architecture {{ARCH}}. Reuse a prototype for operator {{OPERATOR}}.","max_records":3,"max_bytes":12000}
```

Run `python3 tools/plugin.py call gpu-wiki.query --input wiki_request.json` and save its JSON
response under `.atrex_long_horizon/economy_wiki.json`. This standard request is parsed without
launching a bridge agent and searches across DSLs. Read each record's compatibility metadata,
payload and fallback notes before adopting its code. Copy emitted IDs into the short prototype
summary; never reconstruct IDs or treat unrelated fallback samples as matches.
