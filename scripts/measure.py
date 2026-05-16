"""ruri-v3 エンドポイントのコールドスタートを測る。

各ターゲット (TEI 30m / 310m, ST cold 30m / 310m, ST snap 30m / 310m) について:

1. 本計測前に untimed の probe を 1 回打つ。
2. 稼働中コンテナを `modal container stop` で落とす。
3. 初回 POST /embed → かかった時間を `cold` として記録。
4. 続けて 3 回叩いて中央値を `warm` として記録。

URL は `modal.Cls.from_name(...).method.get_web_url()` で取得する。
snapshot 作成/復元の有無は必要なら `modal app logs` を別途見る。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import httpx
import modal
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# `modal container stop` 後にコンテナが完全に落ちるまでの猶予秒数。
DRAIN_SLEEP_SEC = 8

# 初回リクエストはコールドスタート込みで数十秒かかりうるので長めに。
REQUEST_TIMEOUT_SEC = 600

# warm は軽く 10 回だけ測る。
WARM_REQUESTS = 10


TARGETS: list[tuple[str, str, str, str]] = [
    ("tei", "ruri-coldstart-tei", "TEIRuri30m", "serve"),
    ("tei", "ruri-coldstart-tei", "TEIRuri310m", "serve"),
    ("st-cold", "ruri-coldstart-st-cold", "STRuri30m", "web"),
    ("st-cold", "ruri-coldstart-st-cold", "STRuri310m", "web"),
    ("st-snap", "ruri-coldstart-st-snap", "STRuri30m", "web"),
    ("st-snap", "ruri-coldstart-st-snap", "STRuri310m", "web"),
]

text = """世の中で光が当たるのは大抵成功事例で、自分もニュースレターや技術ブログなどをフォローして、有名な企業の事例をよく読んでいる。ただ、その裏には表に出てこない失敗事例が多くあるのは現場で働いているみなさんならご存知の通りで、しかしそういう話はイベントの懇親会で近くの人と話すとかそういう機会でもないとなかなか入手できない。そのような影に葬られがちな話を収集し（匿名化して）扱っている点で、本書は非常に稀有である。

成功というのは運の要素が多くあり、またそのプロダクト特有の事情に左右されたりもするため、なかなか再現できないものである。一方で失敗は、結構限られたパターンに落ち着くことが多い。本書は、そうした失敗事例を20個以上集めており、そのどれもが思い当たる節があったり、自分も同じ現場にいたら同じような判断をしてしまいそう、と感じるものになっている。

例えばロット単位でしか発注できない商品に対して個数単位の予測精度を向上させようと頑張ってしまう事例が紹介されているが、自分も（小売ではないが）そのような予測結果がどう使われるか意識しないモデリングをやってしまったことがある。

最近は AI の性能向上が凄まじく、とりあえず PoC してみよう、とりあえず作って動かしてみよう、というプロダクトアウト的な動きが強くなっているように感じる。個人的にはそれ自体は悪いことではないと思っているが、作って試していろいろやってみても、結局それがユーザーにとって価値がなければ、ビジネスとしては成功しない。

本書にはいろいろな失敗の形が出てくるが、そこから得られる教訓を一言で言うと「ビジネスとして意味のあることをやろう」という話だと理解している。

関係者全員がビジネスとして意味のある方向を向いていれば成功…とまでは行かずとも、想定外の分析結果が出てきた際にも軟着陸できる可能性がある。その一方で、誰かが技術的なことにしか興味を持たなかったり、自分の成果がどう使われるかを想像できなかったり、自分の発言を後から覆せなかったりすると、失敗していくのだろう。また、自分の立場だけで物事を判断してしまうことも、失敗につながりやすい。

本書では、受託分析のように、複数の会社が関わるケースの失敗事例が多く出てくる。会社間でのコミュニケーションがどうしても取りづらかったり、納期がかっちり決まっていたり、クライアントサイドがITに投資していないせいでデータが不完全だったりという構造的な要因が多いのだろうと察せられる。一方、自分の経験の範囲では、事業会社内の施策でも似たような失敗事例や失敗まで行かずとも、これって本当に意味のあることやってるのか？これだけリソースを費やして得られる利益が釣り合っているのか？と首を捻りたくなる事例は少なくない。

ここ数年で一気に高性能な学習済みモデルが使えるようになり、自然言語処理や画像処理は自分でモデルを作る方が今や少数派である。だからこそ、MLエンジニアやデータサイエンティストも、単に技術に詳しいだけではなく、それをどうビジネスとして意味のあるものにするかを考えなければならない時代になっている。

結局のところ、ビジネスとして役に立っているか、ユーザーの課題を解決できているか、会社の利益につながっているか、そういう観点を持ち続けることが重要で、この時代に改めて読むと、とても身につまされる本である。"""
PROBE_PAYLOAD = {"inputs": ["クエリ: " + text]}


def parse_args() -> argparse.Namespace:
    """CLI 引数を解釈する。"""
    parser = argparse.ArgumentParser(
        description="Measure Modal cold start for selected ruri-v3 endpoints."
    )
    parser.add_argument(
        "--pattern",
        action="append",
        choices=sorted({label for label, *_ in TARGETS}),
        help="計測する pattern を絞る。複数回指定可。例: --pattern tei",
    )
    parser.add_argument(
        "--target",
        action="append",
        help=(
            "計測する target を 'pattern/ClassName' 形式で絞る。"
            "複数回指定可。例: --target tei/TEIRuri30m"
        ),
    )
    return parser.parse_args()


def _target_key(target: tuple[str, str, str, str]) -> str:
    label, _app_name, class_name, _web_attr = target
    return f"{label}/{class_name}"


def select_targets(args: argparse.Namespace) -> list[tuple[str, str, str, str]]:
    selected = TARGETS

    if args.pattern:
        pattern_set = set(args.pattern)
        selected = [target for target in selected if target[0] in pattern_set]

    if args.target:
        wanted = set(args.target)
        selected = [target for target in selected if _target_key(target) in wanted]

    return selected


def _resolve_app_id(app_name: str) -> str | None:
    try:
        out = subprocess.run(
            ["modal", "app", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        items = json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return None

    for item in items:
        if item.get("Description") == app_name and item.get("State") == "deployed":
            return item.get("App ID")
    return None


def _running_container_ids(app_id: str) -> list[str]:
    try:
        out = subprocess.run(
            ["modal", "container", "list", "--json", "--app-id", app_id],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        items = json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return []

    ids: list[str] = []
    for item in items:
        for key in ("Container ID", "ID", "id", "container_id"):
            container_id = item.get(key)
            if container_id:
                ids.append(container_id)
                break
    return ids


def force_cold_start(app_name: str) -> int:
    """稼働中コンテナを全部止めて次の request を cold にする。"""
    app_id = _resolve_app_id(app_name)
    if not app_id:
        return 0

    running_ids = _running_container_ids(app_id)
    for container_id in running_ids:
        subprocess.run(
            ["modal", "container", "stop", "-y", container_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
    return len(running_ids)


def lookup_url(app_name: str, class_name: str, web_attr: str) -> str:
    cls = modal.Cls.from_name(app_name, class_name)
    instance = cls()
    method = getattr(instance, web_attr)
    if hasattr(method, "get_web_url"):
        return method.get_web_url()
    return method.web_url


def time_post(
    client: httpx.Client, url: str, headers: dict[str, str]
) -> tuple[float, int]:
    started = time.perf_counter()
    response = client.post(
        url + "/embed",
        json=PROBE_PAYLOAD,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    elapsed = time.perf_counter() - started
    return elapsed, response.status_code


def prewarm_all(
    headers: dict[str, str],
    targets: list[tuple[str, str, str, str]],
) -> None:
    """本計測前に全 endpoint を 1 回ずつ叩く。timing は取らない。"""
    print("=== prewarm pass (untimed) ===", flush=True)
    with httpx.Client() as client:
        for label, app_name, class_name, web_attr in targets:
            print(f"[prewarm] {label}/{class_name}...", flush=True, end=" ")
            try:
                url = lookup_url(app_name, class_name, web_attr)
                response = client.post(
                    url + "/embed",
                    json=PROBE_PAYLOAD,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT_SEC,
                )
                print(f"status={response.status_code}", flush=True)
            except Exception as exc:
                print(f"failed: {exc}", flush=True)
    print("=== prewarm done, starting measurement ===\n", flush=True)


def measure_one(
    label: str,
    app_name: str,
    class_name: str,
    web_attr: str,
    headers: dict[str, str],
) -> dict[str, object]:
    print(f"[{label}/{class_name}] looking up URL...", flush=True)
    try:
        url = lookup_url(app_name, class_name, web_attr)
    except Exception as exc:
        return {"label": label, "class": class_name, "error": f"lookup failed: {exc}"}

    n_stopped = force_cold_start(app_name)
    print(
        f"[{label}/{class_name}] killed {n_stopped} running container(s), "
        f"sleeping {DRAIN_SLEEP_SEC}s...",
        flush=True,
    )
    time.sleep(DRAIN_SLEEP_SEC)

    with httpx.Client() as client:
        print(f"[{label}/{class_name}] POST (cold) {url}/embed", flush=True)
        try:
            cold_s, status = time_post(client, url, headers)
        except Exception as exc:
            return {
                "label": label,
                "class": class_name,
                "error": f"cold POST failed: {exc}",
            }
        if status != 200:
            return {
                "label": label,
                "class": class_name,
                "error": f"cold non-200 status={status}",
                "cold": cold_s,
            }

        warm_samples: list[float] = []
        for _ in range(WARM_REQUESTS):
            try:
                warm_s, status = time_post(client, url, headers)
            except Exception as exc:
                return {
                    "label": label,
                    "class": class_name,
                    "error": f"warm POST failed: {exc}",
                    "cold": cold_s,
                }
            if status != 200:
                return {
                    "label": label,
                    "class": class_name,
                    "error": f"warm non-200 status={status}",
                    "cold": cold_s,
                }
            warm_samples.append(warm_s)

    return {
        "label": label,
        "class": class_name,
        "cold": cold_s,
        "warm_median": statistics.median(warm_samples),
        "warm_min": min(warm_samples),
        "url": url,
    }


def main() -> int:
    args = parse_args()
    targets = select_targets(args)
    if not targets:
        print("No targets matched the given filters.", file=sys.stderr)
        return 1

    load_dotenv()
    key = os.environ.get("MODAL_KEY")
    secret = os.environ.get("MODAL_SECRET")
    if not key or not secret:
        print(
            "MODAL_KEY / MODAL_SECRET must be set (see .env.example)",
            file=sys.stderr,
        )
        return 1
    headers = {"Modal-Key": key, "Modal-Secret": secret}

    prewarm_all(headers, targets)
    results = [measure_one(*target, headers=headers) for target in targets]

    console = Console()
    table = Table(title="ruri-v3 cold-start (L4 / proxy auth / flash-attn-2)")
    table.add_column("pattern")
    table.add_column("class")
    table.add_column("cold (s)", justify="right")
    table.add_column("warm median (s)", justify="right")
    table.add_column("warm min (s)", justify="right")
    table.add_column("note")

    for result in results:
        if "error" in result:
            table.add_row(
                str(result["label"]),
                str(result["class"]),
                f"{result.get('cold', float('nan')):.2f}" if "cold" in result else "-",
                "-",
                "-",
                str(result["error"]),
            )
        else:
            table.add_row(
                str(result["label"]),
                str(result["class"]),
                f"{result['cold']:.2f}",
                f"{result['warm_median']:.3f}",
                f"{result['warm_min']:.3f}",
                "",
            )

    console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
