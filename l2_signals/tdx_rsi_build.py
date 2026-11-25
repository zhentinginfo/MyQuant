"""通达信 RSI 因子构建：core rollup 收盘 → RSI6/12/24 长表 parquet。

公式（通达信自带 RSI，N1=6/N2=12/N3=24 三线）：
    LC := REF(CLOSE, 1);
    RSI1: SMA(MAX(CLOSE-LC, 0), N, 1) / SMA(ABS(CLOSE-LC), N, 1) * 100;

TDX ``SMA(X, N, 1)`` 为威尔德递推 ``Y_t = (X_t + (N-1)*Y_{t-1}) / N``，种子
Y=首个有效 X。无缺口时逐位等价 ``ewm(alpha=1/N, adjust=False)``。首行无 LC →
RSI 无效值；平坦段 0/0 → NaN（通达信无效值）。

停牌缺口（与 alpha158 同族的宽表并集日历）：停牌日不进入递推。LC 取上一有效
收盘，``ewm(..., ignore_na=True)`` 跳过 NaN 并保持状态，输出再把 close 为 NaN
的日子掩成 NaN。复牌当日因此与「K 线里没有这根停牌柱」的 TDX 逐位一致；连续
交易个股同样逐位一致。

用法（vanna312 环境）：
    D:/anaconda3/envs/vanna312/python.exe l2_signals/tdx_rsi_build.py --selftest
    D:/anaconda3/envs/vanna312/python.exe l2_signals/tdx_rsi_build.py
    D:/anaconda3/envs/vanna312/python.exe l2_signals/tdx_rsi_build.py --start 2025-01-01 --ns 6,12,24
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

ROLLUP = "J:/l2_lake/rollup/daily_stock/version=l2-daily-v3/params=a2e3a427af892982d0ce/date=*/part.parquet"
DEFAULT_OUT = "E:/stock_data/l2_research/tdx_rsi_daily.parquet"
DEFAULT_NS = (6, 12, 24)


# ---------------------------------------------------------------- TDX oracle
def _tdx_sma_oracle(values, n):
    """纯 Python TDX SMA(X,N,1)：NaN 不更新状态，输出 NaN，下一有效值接着递推。"""
    y = None
    out = []
    for v in values:
        if v != v:  # NaN
            out.append(float("nan"))
            continue
        y = v if y is None else (v + (n - 1) * y) / n
        out.append(y)
    return out


def _tdx_rsi_oracle(closes, n):
    """LC 跨过 NaN 收盘，取上一有效收盘；停牌日本身输出 NaN。"""
    lc = []
    prev = float("nan")
    up, ab = [], []
    for c in closes:
        if c != c:
            lc.append(float("nan"))
            up.append(float("nan"))
            ab.append(float("nan"))
            continue
        lc.append(prev)
        if prev != prev:
            up.append(float("nan"))
            ab.append(float("nan"))
        else:
            diff = c - prev
            up.append(max(diff, 0.0))
            ab.append(abs(diff))
        prev = c
    su, sa = _tdx_sma_oracle(up, n), _tdx_sma_oracle(ab, n)
    return [float("nan") if a == 0.0 or a != a else 100.0 * u / a for u, a in zip(su, sa)]


def rsi_wide(close: pd.DataFrame, ns) -> dict:
    """宽表（日期×股票）收盘 → {RSI<N>: DataFrame}。

    ``ewm(alpha=1/N, adjust=False, ignore_na=True)`` 在有效点上等于 TDX
    ``SMA(X,N,1)``。LC 用 ``ffill().shift(1)``，使复牌日对上停牌前收盘。
    """
    valid = close.notna()
    lc = close.ffill().shift(1)
    delta = (close - lc).where(valid)
    up = delta.clip(lower=0.0)
    ab = delta.abs()
    out = {}
    for n in ns:
        den = ab.ewm(alpha=1 / n, adjust=False, ignore_na=True).mean()
        num = up.ewm(alpha=1 / n, adjust=False, ignore_na=True).mean()
        rsi = num / den * 100.0
        rsi = rsi.mask(den == 0.0)  # 平坦段 0/0 → NaN
        out[f"RSI{n}"] = rsi.where(valid)  # 停牌日不把上一根 RSI 填过来
    return out


def selftest():
    """黄金值 + 纯 Python oracle 对拍 + 边界性质。"""
    closes = [10, 11, 10.5, 12, 11.8, 12.5, 13, 12, 12.4, 13.5,
              14, 13.2, 13.8, 14.5, 15]
    goldens = {
        6: [np.nan, 100.0, 90.909091, 93.150685, 89.615182, 91.043012,
            91.987310, 73.412364, 75.761689, 81.233789, 83.291346,
            68.806721, 73.028160, 77.324381, 80.048382],
        12: [np.nan, 100.0, 95.652174, 96.193772, 94.481864, 94.832964,
             95.077041, 86.193660, 86.734507, 88.129541, 88.717867,
             81.654084, 82.776121, 84.020004, 84.871356],
        24: [np.nan, 100.0, 97.872340, 98.005204, 97.161058, 97.247639,
             97.308813, 92.995245, 93.122496, 93.463252, 93.613340,
             90.157215, 90.433619, 90.749851, 90.972279],
    }
    wide = pd.DataFrame({"s": closes})
    got = rsi_wide(wide, goldens.keys())
    for n, golden in goldens.items():
        np.testing.assert_allclose(got[f"RSI{n}"]["s"].to_numpy(), golden,
                                   rtol=0, atol=5e-7)
    # 随机序列（无内部 NaN）oracle 对拍
    rng = np.random.RandomState(7)
    m = pd.DataFrame(rng.randn(400, 5).cumsum(axis=0) + 100.0)
    got = rsi_wide(m, (6, 12, 24))
    for j in range(m.shape[1]):
        col = m.iloc[:, j].tolist()
        for n in (6, 12, 24):
            np.testing.assert_allclose(
                got[f"RSI{n}"].iloc[:, j].to_numpy(), _tdx_rsi_oracle(col, n),
                rtol=0, atol=1e-9)
    # 停牌缺口：跳过 NaN 柱，LC=上一有效收盘，复牌日继续递推（手算 N=6）
    gapped = [10.0, 11.0, np.nan, 12.0, 11.5, 12.2]
    gap_golden = [np.nan, 100.0, np.nan, 100.0, 90.9090909090909, 92.11356466876971]
    gap_got = rsi_wide(pd.DataFrame({"s": gapped}), (6,))["RSI6"]["s"].to_numpy()
    np.testing.assert_allclose(gap_got, gap_golden, rtol=0, atol=1e-9)
    np.testing.assert_allclose(gap_got, _tdx_rsi_oracle(gapped, 6), rtol=0, atol=1e-9)
    # 性质：单边上行=100；平坦=NaN；首行必 NaN
    up_only = rsi_wide(pd.DataFrame({"s": np.arange(1.0, 21.0)}), (6,))["RSI6"]["s"]
    assert np.isnan(up_only.iloc[0]) and (up_only.iloc[1:] == 100.0).all()
    flat = rsi_wide(pd.DataFrame({"s": np.full(20, 5.0)}), (6,))["RSI6"]["s"]
    assert flat.isna().all()
    print("[selftest] goldens OK; oracle(6/12/24) OK; halt-gap OK; monotone/flat OK")


# ---------------------------------------------------------------- 因子计算
def compute_rsi(chunk: pd.DataFrame, ns) -> pd.DataFrame:
    """一个股票块（长表 d×stock×close）→ 长表 d/stock_code + RSI 因子。"""
    dates = np.sort(chunk["d"].unique())
    stocks = chunk["stock_code"].unique()
    close = chunk.pivot(index="d", columns="stock_code",
                        values="close_px").reindex(dates)[stocks]
    out = pd.DataFrame({
        "d": np.repeat(dates, len(stocks)),
        "stock_code": np.tile(stocks, len(dates)),
    })
    arrs = {k: v.to_numpy(dtype=np.float32).ravel()
            for k, v in rsi_wide(close, ns).items()}
    out = pd.concat([out, pd.DataFrame(arrs, index=out.index)], axis=1)
    out = out[out[[f"RSI{n}" for n in ns]].notna().any(axis=1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--start", default="2025-01-01", help="输出日期下界（滚动仍用全历史）")
    ap.add_argument("--ns", default=",".join(map(str, DEFAULT_NS)), help="逗号分隔周期，如 6,12,24")
    ap.add_argument("--chunk", type=int, default=800, help="每块股票数")
    ap.add_argument("--limit-chunks", type=int, default=0, help=">0 时试跑指定块数")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    selftest()
    ns = tuple(sorted({int(x) for x in args.ns.split(",") if x.strip()}))
    assert all(n >= 1 for n in ns), "周期必须 >= 1"

    t0 = time.time()
    lf = pl.scan_parquet(ROLLUP, hive_partitioning=False).select(
        ["stock_code", "d", "close_px"]).collect()
    df = lf.to_pandas()
    df["d"] = pd.to_datetime(df["d"]).dt.date
    df = df.sort_values(["stock_code", "d"])
    stocks = sorted(df["stock_code"].unique())
    print(f"[load] {len(df):,} 行 × {len(stocks)} 票  {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = None
    start = pd.to_datetime(args.start).date()
    n_done = 0
    for ci in range(0, len(stocks), args.chunk):
        block = stocks[ci: ci + args.chunk]
        sub = df[df["stock_code"].isin(block)]
        res = compute_rsi(sub, ns)
        res = res[res["d"] >= start]
        if len(res) == 0:
            continue
        res = res.sort_values(["d", "stock_code"])
        table = pa.Table.from_pandas(res, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.out, table.schema, compression="zstd")
        writer.write_table(table)
        n_done += 1
        print(f"[chunk {n_done}] {block[0]}..{block[-1]}  rows={len(res):,}  {time.time()-t0:.0f}s", flush=True)
        if args.limit_chunks and n_done >= args.limit_chunks:
            break
    if writer:
        writer.close()
        n = pq.ParquetFile(args.out).metadata.num_rows
        print(f"[done] {args.out}  rows={n:,}  {time.time()-t0:.0f}s")
    else:
        print("[done] 无输出行")


if __name__ == "__main__":
    sys.exit(main())
