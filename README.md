# The Clock Inside the AI

**Do language models keep the days of the week on a circle?** We checked inside Sarvam-1, a 2-billion-parameter model for Indian languages. Yes: Monday to Sunday form a ring in its internal numbers, and Hindi weekday names land on the same ring as the English ones. Turning that ring with a "dial" changes what the model says the next day is.

**▶ Live demo:** https://samkoduru.github.io/clock-inside-the-ai/clock.html (replay mode: the dial shows recorded runs). To drive the dial with the real model, run it locally (see below).

**Companion demo:** [Watch the AI Pretend to Think](https://samkoduru.github.io/watch-ai-pretend-to-think/theater.html)

![status](https://img.shields.io/badge/model-Sarvam--1%202B-orange) ![compute](https://img.shields.io/badge/compute-laptop%20(M3%20Pro)%20%240-brightgreen) ![license](https://img.shields.io/badge/code-MIT-blue)

## Findings

Every number below is produced by the scripts in this repo. Layer-wise results come from Sarvam-1 (28 layers), last token of each name, averaged over 8 sentence templates.

| Concept | Best ring score (1 = perfect circle) | Layer | p (labels shuffled, corrected for picking the best layer) |
|---|---|---|---|
| English weekdays | 0.98 | 20 | 0.0028 (14 of 5,040 orderings score as high) |
| Hindi weekdays | 0.99 | 20–21 | 0.0028 |
| English months | 1.00 | 20 | < 0.0001 |
| Hindi months | 0.95 | 18 | < 0.0001 |
| Tamil weekdays | 0.58 | — | 0.70 (no ring) |

- **Cross-language alignment.** Fit the ring on English, then measure where Hindi names land. Alignment is near zero in layers 1–8 (0.09–0.54) and reaches 0.84–1.00 from layer 14 onward (dip at layer 13). Shuffled-label p = 0.0002.
- **Random-word control.** Thirty random sets of seven ordinary nouns, same procedure: best-layer ring score averages **0.71** (max 0.92). A high ring score alone is not evidence; the label-shuffle test is what carries the claim.
- **The dial.** Replace the model's internal representation of "Friday" with a point partway around the ring, then read its answer to "tomorrow is ___". At layer 12, with three Fourier harmonics on the ring:

  | Harmonics used | Accuracy (chance 0.14) |
  |---|---|
  | 1 | 0.26 |
  | 2 | 0.60 |
  | 3 | 0.90 |
  | random subspace of the same size | 0.00 |

  Probability mass on the correct neighbouring days: ring path 0.69, straight line between days 0.63, scrambled-order control 0.27.
- **Tamil has no ring.** Every Tamil weekday name ends in the same word, கிழமை ("day"). That may be why the last token carries little day information. Untested.

## Try it

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). Models download on first run (Sarvam-1 is about 4.5 GB).

```bash
git clone https://github.com/Samkoduru/clock-inside-the-ai
cd clock-inside-the-ai
uv sync

# 1. Replay mode, no model needed:
uv run python serve.py --no-live            # then open http://127.0.0.1:8000/clock.html

# 2. Live mode, the dial calls the real model on your machine:
uv run python calendar_spike.py            # ~3 min on Apple Silicon: caches activations to cache/
uv run python serve.py                     # the dial badge turns green: "live · model running on this laptop"
```

Run from the repo root. The page reads `web/data.json` (committed) for everything except the live dial.

## Reproduce the numbers

```bash
uv run python calendar_spike.py            # activations, per-layer ring scores, cross-language alignment
uv run python rotate_spike.py              # early intervention test (rotation vs random plane)
uv run python dial_spike.py                # harmonic dial accuracy by layer
uv run python export_data.py               # corrected statistics, random-word control, dial sweep -> web/data.json
uv run pytest                              # geometry unit tests (ring score invariance, dial reproduces class means)
```

## How it works

1. **Collect.** Put each day or month name into short sentences ("Today is ___", "The meeting is on ___") in English, Hindi and Tamil, run the model, and average the hidden states at the name's last token.
2. **Find the ring.** At each layer, take the top two principal components of the seven class means and measure how well the angles follow the true weekday order. Compare with every label permutation (exhaustive for 7 items) and take the best layer, so the p-value accounts for choosing among 28 layers.
3. **Cross-language.** Fit the ring on English, project the Hindi class means, and measure how far each one is from its English twin (up to a single global rotation).
4. **Dial.** Decompose the ring into Fourier harmonics over the seven days, reconstruct a continuous position on it, and patch the model's residual stream at the day token. Controls: random subspace of the same size, and a scrambled day order.

## Limits

- One model (Sarvam-1, 2B), seven items per concept, one sentence set per language. We have not tested other models.
- Layers were picked after looking at results. The p-values correct for this; the dial accuracy does not.
- The dial gets the answer right 90% of the time with three harmonics, but the ring path is only modestly better than a straight line (0.69 vs 0.63 probability mass). The scrambled control is clearly worse (0.27), so the ring structure matters, not just "moving the name."
- Tamil is an open question. The explanation above is a hypothesis.

## Layout

```
calendar_spike.py   collect activations, per-layer ring scores, cross-language alignment
rotate_spike.py     rotation intervention vs random-plane control
dial_spike.py       harmonic ring model and dial accuracy
export_data.py      all corrected statistics and the dial sweep -> web/data.json
serve.py            static server plus live /api/dial (uses Sarvam-1 on your machine)
web/clock.html      the interactive walkthrough (scroll-driven, three.js), deployed by .github/workflows/pages.yml
tests/              geometry unit tests
```

## Credits and licences

- Idea and framing follow public work on circular and Fourier structure in language models. This is an independent small-scale replication, not code or results from any group.
- **Sarvam-1** is distributed under the Sarvam AI Research License. Use here is for research and demonstration; commercial use needs Sarvam's written permission. We have no affiliation with Sarvam AI and this project is not endorsed by them.
- Code in this repository is MIT licensed (see `LICENSE`). `web/vendor/three.module.min.js` is three.js, MIT.
