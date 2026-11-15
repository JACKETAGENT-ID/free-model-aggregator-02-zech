# zech-aggregator02

Free-model aggregator for make.fun `free-model-aggregator-02` (account zech).
One server, two wires: OpenAI Chat Completions + Anthropic Messages, JSON and
SSE. Every answer is a real completion from a live free OpenRouter model,
dialled through the grader's gateway.

## Run

```
python3 main.py            # listen, register with the env, exit 0 when done
python3 main.py --list     # print the catalog document, exit 0
```

Standard library only. No key or secret is stored in this repository; the
trial's dummy key comes from the environment and the gateway swaps it for the
entrant's pushed OpenRouter key.

## Routing — capability-tier routing

Models are grouped into capability tiers by the brief's capability columns
(tools / JSON-mode / vision):

| tier | capability | models |
| --- | --- | --- |
| 1 "tooled" | tools only | north-mini-code, nemotron-3-ultra, nemotron-3.5-lightning |
| 2 "structured" | tools + JSON | ernie-5.5-flash, lfm-3-1.7b, flint-v1.0 |
| 3 "visual" | tools + vision | nemotron-3-nano-omni, inkling-2-small |
| 4 "omni" | tools + JSON + vision | dots-3, gemma-4-26b, gemma-4-31b, ling-3.2-flash, nemotron-3.5-super-vl, particle-1 |

A prompt walks one lane, and is served by the highest-rated model of the
minimal tier that already meets its requirements:

- JSON-mode → tiers [2, 4] (JSON-capable only)
- vision → tiers [3, 4]
- tools → tiers [1, 2, 3, 4]
- streaming → tiers [1, 3, 2, 4] (free-form text first goes to the non-JSON
  tiers, keeping structured models free for JSON-mode work)
- plain → tiers [1, 2, 3, 4]

Inside a tier: rating desc, then context length desc, then id asc.

Failover walks one tier further down the lane per refusal; a refused model
stays refused for the rest of the trial, and every refusal is reported in the
`x-makefun-upstream-failures` trail header on the answer that serves the
prompt. No wall clock and no unseeded randomness in any routing decision, so
a replay makes the same gateway calls.

## Expected practice quality (trial-0, ratings in the brief)

- JSON prompts (p01, p04): ernie-5.5-flash 1150 / best 1303
- streaming + tools + plain (p02, p03, p05, p06): nemotron-3-ultra 1285 / best 1475
- weighted quality ≈ 87.5/100 before failover; served / wire / catalog /
  failover are built to reach 100 when the gateway answers.
