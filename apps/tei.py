"""TEI (HF Text Embeddings Inference) を Modal で公開する。

Modal 上で Rust バイナリ (`text-embeddings-router`) を起動し、コンテナ内で
localhost:8000 にぶら下げ、`@modal.web_server` でその port を外に proxy する。

公式 example (modal-labs/modal-examples/06_gpu_and_ml/embeddings/text_embeddings_inference.py)
を踏襲。違いは:
- TEI image を sm_89 (L4) 用 `89-1.9.3` に更新
- 公開方法を `@modal.method` ではなく `@modal.web_server` + proxy auth に
- 重みを image bake せず Modal Volume にキャッシュ
"""

import socket
import subprocess
import time

import modal

from apps._common import (
    GPU,
    HF_CACHE_DIR,
    HF_HUB_CACHE_DIR,
    HF_ENV,
    MODEL_30M,
    MODEL_310M,
    MODEL_VOLUME,
    PYTHON_VERSION,
    SCALEDOWN,
)

# App は「デプロイ単位の名前空間」。同じ App 名で再デプロイすると上書きになる。
app = modal.App("ruri-coldstart-tei")

# TEI 公式の事前ビルド image。L4 (sm_89, Ada Lovelace) 向け。
# 他に: `cpu-1.9.3` / `86-1.9.3` (A10G) / `hopper-1.9.3` (H100) など。
TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:89-1.9.3"
PORT = 8000


def spawn_server(model_id: str) -> subprocess.Popen:
    """HF cache を参照しながら TEI バイナリを起動する。"""
    proc = subprocess.Popen(
        [
            "text-embeddings-router",
            "--model-id", model_id,
            "--huggingface-hub-cache", HF_HUB_CACHE_DIR,
            "--port", str(PORT),
            # 0.0.0.0 で listen させないと web_server proxy から見えない
            "--hostname", "0.0.0.0",
        ]
    )
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            return proc
        except (OSError, ConnectionRefusedError):
            # サーバが先に死んでいたら永遠にループしないよう打ち切り
            if proc.poll() is not None:
                raise RuntimeError(
                    f"text-embeddings-router exited with code {proc.returncode}"
                )
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("text-embeddings-router did not become ready within 300s")


# Image.run_function に渡す関数は **モジュールトップで定義** する必要がある
# （モジュール grab で再構成するため、ローカル変数キャプチャは避ける）。
# モデル毎に別関数を用意して、それぞれ別 image としてビルドする。
def _prefetch_30m():
    spawn_server(MODEL_30M).terminate()


def _prefetch_310m():
    spawn_server(MODEL_310M).terminate()


def _make_image(prefetch_fn) -> modal.Image:
    """TEI 用 image を組み立てる。

    ステップ:
    1. TEI 公式 image を起点にする (`from_registry`)。
    2. `ENTRYPOINT []` でデフォルト ENTRYPOINT を消す。これをやらないと
       Modal がコンテナを起動するとき TEI 起動コマンドが先に走ってしまう。
    3. HF キャッシュ系の env を設定する。
    4. `run_function` で **イメージビルド時に** TEI を一度起動し、Hub repo id から
       `/cache/hf/hub` に必要ファイルを揃える。runtime も同じ repo id を渡し、
       HF cache の速い経路を使う。
    5. `add_local_python_source("apps")` でローカルの `apps/` パッケージを
       コンテナの Python パスに追加（`from apps._common import ...` を
       コンテナ側でも解決できるようにするため。Modal 1.x では明示が必要）。
    """
    return (
        modal.Image.from_registry(TEI_IMAGE, add_python=PYTHON_VERSION)
        .dockerfile_commands("ENTRYPOINT []")
        .env(HF_ENV)
        # `copy=True` & `run_function` より **前** に置く。
        # entry script (`tei.py`) が module-level で `from apps._common import ...`
        # しているので、`run_function` が走る時点で `apps` パッケージがイメージに
        # 入っていないと `ModuleNotFoundError` で落ちる。
        .add_local_python_source("apps", copy=True)
        .run_function(
            prefetch_fn,
            gpu=GPU,                                 # build 時にも GPU が必要
            volumes={HF_CACHE_DIR: MODEL_VOLUME},    # build 時から Volume をマウント
        )
    )


# ---- 30m ----------------------------------------------------------------
# `@app.cls(...)` で **クラスを Modal 関数化** する。`@modal.enter` で起動時、
# `@modal.exit` で終了時のフックを定義できる。
# - max_containers=1: 計測の安定のため並列を抑制（本番では外す）
# - volumes=...     : runtime にも Volume をマウント（build 時と同じ場所に）
@app.cls(
    image=_make_image(_prefetch_30m),
    gpu=GPU,
    scaledown_window=SCALEDOWN,
    max_containers=1,
    volumes={HF_CACHE_DIR: MODEL_VOLUME},
)
class TEIRuri30m:
    @modal.enter()
    def boot(self):
        # コンテナごとに 1 回だけ呼ばれる。リクエストが届く前に走る。
        self.proc = spawn_server(MODEL_30M)

    @modal.exit()
    def shutdown(self):
        self.proc.terminate()

    # `@modal.web_server(port)` は「コンテナ内の port を外部 HTTP に晒す」決定的に
    # シンプルな公開方法。FastAPI 的な使い方なら asgi_app / fastapi_endpoint を使う。
    # - requires_proxy_auth=True: Modal-Key / Modal-Secret ヘッダ必須にする
    # - startup_timeout       : `@modal.enter` を含めて何秒以内に listen するか
    @modal.web_server(port=PORT, requires_proxy_auth=True, startup_timeout=300)
    def serve(self):
        # 関数本体はダミー。Modal は `@modal.enter` 完了後、port を listen して
        # いるコンテナにリクエストを proxy するだけ。
        pass


# ---- 310m ---------------------------------------------------------------
@app.cls(
    image=_make_image(_prefetch_310m),
    gpu=GPU,
    scaledown_window=SCALEDOWN,
    max_containers=1,
    volumes={HF_CACHE_DIR: MODEL_VOLUME},
)
class TEIRuri310m:
    @modal.enter()
    def boot(self):
        self.proc = spawn_server(MODEL_310M)

    @modal.exit()
    def shutdown(self):
        self.proc.terminate()

    @modal.web_server(port=PORT, requires_proxy_auth=True, startup_timeout=300)
    def serve(self):
        pass
