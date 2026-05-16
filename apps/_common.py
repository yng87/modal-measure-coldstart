"""共通定数。

Modal の用語ミニ解説:
- App      : デプロイ単位。`modal deploy <file>` 時に作られる名前空間。
- Image    : コンテナイメージの定義。`from_registry` / `debian_slim` から始めて
             `.uv_pip_install(...)` などのメソッドを **チェーン** していく。
             各チェーン段階は遅延評価で、実際のビルドはデプロイ時に走る。
- Volume   : 永続ストレージ。複数の App / コンテナ間で共有可能。HF キャッシュを
             ここに置けば、イメージを再ビルドしても重みは消えない。
- Cls      : `@app.cls(...)` でデコレートしたクラス。`@modal.enter` / `@modal.exit`
             でコンテナの起動/終了に処理を仕込める。
"""

import os

import modal

# L4 = sm_89 (Ada Lovelace, 24GB)。30m / 310m には十分でコストも低い。
# Modal の GPU 指定は文字列。"A10G" / "A100" / "H100" / "T4" など。
GPU = "L4"

# scaledown_window: 最後のリクエストから何秒で空きコンテナを落とすか。
# 短いほどコストが低い代わりに次回リクエストが cold start になりやすい。
# default は 60 秒。計測では既定で十分。
SCALEDOWN = 60

# flash-attn の wheel が cp311 ビルドなので Python も 3.11 で揃える。
PYTHON_VERSION = "3.11"

# flash-attn の wheel ファイル名にエンコードされた torch バージョンと一致させる。
TORCH_VERSION = "2.6.0"

# `from_name(..., create_if_missing=True)` は **遅延参照** で、ローカルでは
# 実際の作成は走らない。デプロイ時に Modal 側で作られ、無ければ自動生成される。
# 同じ名前を別 App で参照すれば内容を共有できる。
MODEL_VOLUME = modal.Volume.from_name("ruri-weights", create_if_missing=True)

# Volume をマウントする先。build / runtime ともここを共有する。
HF_CACHE_DIR = "/cache/hf"
# HF Hub 自体の blob/snapshot cache
HF_HUB_CACHE_DIR = f"{HF_CACHE_DIR}/hub"
# transformers が見に行く cache
TRANSFORMERS_CACHE_DIR = f"{HF_CACHE_DIR}/transformers"
# sentence-transformers 独自 cache
SENTENCE_TRANSFORMERS_HOME_DIR = f"{HF_CACHE_DIR}/sentence-transformers"
# runtime で直接読む「展開済みモデル」の置き場所
MODEL_EXPORT_ROOT = f"{HF_CACHE_DIR}/models"
MODEL_30M_LOCAL_DIR = f"{MODEL_EXPORT_ROOT}/ruri-v3-30m"
MODEL_310M_LOCAL_DIR = f"{MODEL_EXPORT_ROOT}/ruri-v3-310m"

# HF 系ライブラリは下記 env 変数を見てキャッシュ場所を決める。
HF_ENV = {
    "HF_HOME": HF_CACHE_DIR,
    "HF_HUB_CACHE": HF_HUB_CACHE_DIR,
    "HUGGINGFACE_HUB_CACHE": HF_HUB_CACHE_DIR,  # TEI バイナリも見る
    "TRANSFORMERS_CACHE": TRANSFORMERS_CACHE_DIR,
    "SENTENCE_TRANSFORMERS_HOME": SENTENCE_TRANSFORMERS_HOME_DIR,
}

# Modal 公式 example (https://modal.com/docs/examples/install_flash_attn) と同じ pin 方式。
# wheel ファイル名が「cu12 / torch2.6 / cxx11abiFALSE / cp311」という互換情報を
# エンコードしているので、torch のバージョンと wheel URL は **同じ uv_pip_install
# レイヤ** に並べて宣言する（解決を 1 度で完結させ ABI 不整合を避けるため）。
# wheel が消えた場合は ↓ を新しいリリースに差し替え、TORCH_VERSION も合わせる。
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

MODEL_30M = "cl-nagoya/ruri-v3-30m"
MODEL_310M = "cl-nagoya/ruri-v3-310m"


def prefetch_model_to_volume(repo_id: str, local_dir: str) -> None:
    """HF Hub から Volume 上の固定パスへモデル一式を展開する。"""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        cache_dir=HF_HUB_CACHE_DIR,
    )
    # mounted Volume への変更を後続コンテナから確実に見えるようにする
    MODEL_VOLUME.commit()


# memory snapshot を試す/試さないをデプロイ時に env で切り替えるための仕組み。
# `ENABLE_SNAPSHOT=1 modal deploy ...` で「snap モード」のアプリ名にして
# 別アプリとして同居させ、計測スクリプトが両方を順に叩く設計。
SNAPSHOT = os.environ.get("ENABLE_SNAPSHOT") == "1"
APP_SUFFIX = "-snap" if SNAPSHOT else "-cold"
