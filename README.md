# modal-measure-coldstart

Modal 上で ruri-v3 の cold start を TEI / Sentence Transformers / snapshot on-off で測る実験用リポジトリ。

## Build

```bash
cp .env.example .env
uv sync
uv run modal deploy apps/tei.py
uv run modal deploy apps/transformers_app.py
ENABLE_SNAPSHOT=1 uv run modal deploy apps/transformers_app.py
```

## Measure

```bash
uv run python scripts/measure.py
```
