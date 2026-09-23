#!/usr/bin/env python3
"""Smoke-test a served model over the OpenAI-compatible API: identity, reasoning, coding, long-context recall."""
import json, sys, time, urllib.request
URL = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.8-Flash-Next-hle96k"
QUESTIONS = [
    ("自我介绍", "你好，用一句话介绍你自己。"),
    ("数学推理", "一个正整数 n，满足 n 除以 7 余 3，除以 11 余 5，求最小的 n。请给出推导。"),
    ("代码", "写一个 Python 函数，用二分查找在有序列表中找目标值的插入位置。"),
    ("常识问答", "简要说明光合作用的两个阶段及各自的产物。"),
    ("英文", "Explain in two sentences why transformers replaced RNNs for sequence modeling."),
]
for name, q in QUESTIONS:
    body = {"model": MODEL, "messages": [{"role": "user", "content": q}], "temperature": 0.3, "max_tokens": 1024}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    content = (m.get("content") or "").strip()
    think = (m.get("reasoning_content") or "").strip()
    u = d.get("usage", {})
    print(f"\n{'=' * 12} {name}  ({time.time() - t0:.1f}s, {u.get('completion_tokens')} tok)")
    print(f"Q: {q}")
    if think:
        print(f"[思考 {len(think)} 字，前 100]: {think[:100]}")
    print(f"A: {content[:600] if content else '(空，思考未结束)'}")
