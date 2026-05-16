"""sentence-transformers + flash-attn-2 を FastAPI で公開する。

TEI 側と違って Python プロセスが直接 inference を担う。コンテナ起動時に
sentence-transformers モデルをメモリにロードし、リクエストごとに `model.encode`
を呼ぶ。FastAPI はそれを HTTP に乗せるだけ。

`enable_memory_snapshot` は env `ENABLE_SNAPSHOT=1` で切り替える設計。
- snapshot off → アプリ名 `ruri-coldstart-st-cold`
- snapshot on  → アプリ名 `ruri-coldstart-st-snap`
別アプリとして同居させ、計測スクリプトが両方を順に叩く。
"""

import modal

from apps._common import (
    APP_SUFFIX,
    FLASH_ATTN_WHEEL,
    GPU,
    HF_CACHE_DIR,
    HF_ENV,
    MODEL_30M,
    MODEL_30M_LOCAL_DIR,
    MODEL_310M,
    MODEL_310M_LOCAL_DIR,
    MODEL_VOLUME,
    PYTHON_VERSION,
    SCALEDOWN,
    SNAPSHOT,
    TORCH_VERSION,
    prefetch_model_to_volume,
)

# SNAPSHOT の有無でアプリ名を変えて、両方を同じワークスペースに同居させる。
app = modal.App(f"ruri-coldstart-st{APP_SUFFIX}")


# Image.run_function に渡す関数。モジュールトップで定義する。
# HF Hub から repo snapshot を落とし、runtime ではそのローカルパスだけを読む。
def _prefetch_30m():
    prefetch_model_to_volume(MODEL_30M, MODEL_30M_LOCAL_DIR)


def _prefetch_310m():
    prefetch_model_to_volume(MODEL_310M, MODEL_310M_LOCAL_DIR)


def _make_image(prefetch_fn) -> modal.Image:
    """sentence-transformers + flash-attn-2 入りの image を組み立てる。

    ★ 一番のポイント:
    `torch==X.Y.Z` と flash-attn の wheel URL を **同じ uv_pip_install レイヤ** に
    並べる。理由は wheel ファイル名 (`+cu12torch2.6cxx11abiFALSE-cp311`) が
    互換情報をエンコードしていて、ここで torch 版がブレると ABI 不整合で
    `ImportError` になるため。Modal 公式 example と同じやり方。
    """
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .uv_pip_install(
            f"torch=={TORCH_VERSION}",
            FLASH_ATTN_WHEEL,
            "sentence-transformers>=3.3",
            "transformers>=4.48",                 # ModernBERT-Ja サポート
            "fastapi[standard]>=0.115",
        )
        .env(HF_ENV | {"ENABLE_SNAPSHOT": "1" if SNAPSHOT else "0"})
        # entry script の module-level import (`from apps._common import ...`) を
        # `run_function` 内でも解決させるため、`copy=True` で前に置く。
        .add_local_python_source("apps", copy=True)
        .run_function(prefetch_fn, volumes={HF_CACHE_DIR: MODEL_VOLUME})
    )


# `@modal.enter(snap=True)` は class が `enable_memory_snapshot=True` で
# ないと弾かれるので、SNAPSHOT のときだけ snap=True を渡す。
ENTER_KWARGS = {"snap": True} if SNAPSHOT else {}


def _cls_kwargs(image: modal.Image) -> dict:
    """`@app.cls(...)` に渡す共通キーワード引数。SNAPSHOT で分岐。"""
    kwargs = dict(
        image=image,
        gpu=GPU,
        scaledown_window=SCALEDOWN,
        max_containers=1,
        volumes={HF_CACHE_DIR: MODEL_VOLUME},
        # CPU snapshot は GA、GPU snapshot は alpha。GPU 対応は ↓ の experimental_options が必要。
        enable_memory_snapshot=SNAPSHOT,
    )
    if SNAPSHOT:
        kwargs["experimental_options"] = {"enable_gpu_snapshot": True}
    return kwargs


def _build_fastapi(model):
    """`/embed` エンドポイントだけ持つ最小 FastAPI app を組み立てる。"""
    from fastapi import FastAPI

    api = FastAPI()

    @api.post("/embed")
    def embed(req: dict):
        inputs = req.get("inputs", [])
        if not isinstance(inputs, list):
            inputs = [inputs]
        # normalize_embeddings=True で L2 正規化済みベクトルを返す（cosine 類似度用）
        embs = model.encode(inputs, normalize_embeddings=True).tolist()
        return {"embeddings": embs, "dim": len(embs[0]) if embs else 0}

    return api


def _warmup_model(model):
    """初回 encode の lazy init を軽く消化しておく。"""
    model.encode(["ウォームアップ"], normalize_embeddings=True)


# ---- 30m ----------------------------------------------------------------
@app.cls(**_cls_kwargs(_make_image(_prefetch_30m)))
class STRuri30m:
    # `@modal.enter`: コンテナ起動時に 1 回だけ走るフック。リクエスト処理前に
    # モデルをメモリにロードしておくのが定石。
    # snap=True (snapshot 有効時) を渡すと、このメソッドの実行後の状態が
    # **メモリスナップショット** として保存され、次回以降のコンテナはそこから
    # 復元されるので大幅に cold start が短くなる（GPU 復元は alpha）。
    @modal.enter(**ENTER_KWARGS)
    def load(self):
        import os
        # 重みは Volume に揃っているので HF Hub への metadata round trip は
        # 不要。`HF_HUB_OFFLINE=1` で API 呼び出しごとスキップして cold start を
        # 短くする（"unauthenticated requests" 警告も消える）。
        # build 時の prefetch では DL が必要なのでここで runtime 側だけ立てる。
        os.environ["HF_HUB_OFFLINE"] = "1"

        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL_30M_LOCAL_DIR,
            device="cuda",
            model_kwargs={
                # ModernBERT-Ja を flash-attn-2 で動かす指定
                "attn_implementation": "flash_attention_2",
                # flash-attn-2 は fp16/bf16 必須。L4 は bf16 が無難
                "torch_dtype": torch.bfloat16,
            },
            local_files_only=True,
            trust_remote_code=True,
        )
        _warmup_model(self.model)

    # `@modal.asgi_app` は ASGI app（ここでは FastAPI）を Modal の HTTP 入口に
    # 結びつける。リクエストごとに「この関数の戻り値の app に投げる」感覚。
    # asgi_app() は **コンテナ起動後に 1 度だけ呼ばれて FastAPI app を組み立てる**
    # のでロード済みの self.model をそのまま使える。
    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        return _build_fastapi(self.model)


# ---- 310m ---------------------------------------------------------------
@app.cls(**_cls_kwargs(_make_image(_prefetch_310m)))
class STRuri310m:
    @modal.enter(**ENTER_KWARGS)
    def load(self):
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"

        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL_310M_LOCAL_DIR,
            device="cuda",
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16,
            },
            local_files_only=True,
            trust_remote_code=True,
        )
        _warmup_model(self.model)

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        return _build_fastapi(self.model)
