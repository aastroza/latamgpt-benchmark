# Dataset citations and comparison notes

Last reviewed: 2026-04-22

This document records the official citation text exposed on the Hugging Face dataset cards and the closest comparison protocol we can currently run from this repository.

Sources:

- [Trueque dataset card](https://huggingface.co/datasets/latam-gpt/Trueque-Benchmark-beta-0.1)
- [CHOCLO dataset card](https://huggingface.co/datasets/latam-gpt/CHOCLO)

## Citation policy for this repository

If we publish benchmark results, blog posts, slides, reports, or model cards based on these datasets, we should cite both the dataset cards and use the benchmark names consistently:

- Trueque: `Trueque: A human-reviewed collaborative benchmark for Latin American knowledge and culture`
- CHOCLO: `CHOCLO: Latin American Cultural Knowledge Benchmark`

## Official citations

### Trueque

The dataset card provides this citation:

```bibtex
@software{Trueque_benchmark_beta_0.1,
  title={Trueque: A human-reviewed collaborative benchmark for Latin American knowledge and culture},
  author={Fuentes, Gonzalo and Arriagada, Alexandra and Henriquez, Clemente and García, M. Alexandra and LatamGPT Team},
  year={2026},
  url={https://huggingface.co/latam-gpt/Trueque-Benchmark-beta-0.1}
}
```

### CHOCLO

The dataset card provides this citation:

```bibtex
@dataset{choclo2026,
  title={CHOCLO: Latin American Cultural Knowledge Benchmark},
  author={Del Solar, Bianca and Carvallo, Andrés and Soto, Álvaro},
  year={2026},
  publisher={Hugging Face},
  note={Developed as part of doctoral research under the supervision of Álvaro Soto}
}
```

## Can we compare against the published results?

### Trueque

Not directly, at least not yet.

What the card says:

- this is still a beta release
- baseline results are still pending
- the evaluation setup is still being adjusted
- the current baseline model used in automated evaluation experiments is `Qwen2.5-Instruct (72B)`

Implication:

- There are no stable headline benchmark numbers on the card that we can fairly compare against today.
- The card does not expose enough detail to reproduce the exact current evaluation pipeline from this repository alone.

Closest run from this repository today:

```bash
uv run latamgpt-benchmark \
  --dataset trueque \
  --split train \
  --disable-judge
```

Why this is only approximate:

- our judge is not the card's stated `Qwen2.5-Instruct (72B)`
- the card says the evaluation pipeline is still under revision
- the dataset card does not publish final baseline scores yet

If we want a closer Trueque comparison later, the next step is to support the exact judge model they mention and mirror their final public evaluation protocol once it is released.

### CHOCLO

Partially, but not as an apples-to-apples comparison with the current codebase.

What the card publishes:

- full-dataset aggregate results
- model list: `GPT-4o Mini`, `GPT-3.5 Turbo`, `GPT-5 Mini`, `Mistral`, `DeepSeek`
- a hybrid evaluation framework combining:
  - lexical similarity
  - embedding-based semantic similarity
  - LLM-as-a-judge
- a final average score computed from those components

Implication:

- We can match the dataset and some of the candidate model IDs.
- We cannot claim direct comparability until we replicate their exact metric stack, especially the embedding-based semantic score and the specific LLM judge setup.

Closest run from this repository today for the OpenAI subset:

```bash
uv run latamgpt-benchmark \
  --dataset choclo \
  --split train \
  --model openai:gpt-4o-mini \
  --model openai:gpt-3.5-turbo \
  --model openai:gpt-5-mini
```

Important caveat:

- This is only a candidate-model comparison on the same benchmark questions.
- It is not a faithful reproduction of the published CHOCLO scores.

Why not:

- our deterministic metrics are `normalized_exact_match`, `token_f1`, and `char_similarity`
- the card reports lexical, embedding, and LLM judge scores
- our current judge prompt and judge model are different from the unpublished CHOCLO setup

## Recommended wording for future reports

Use wording like this:

"We evaluate on CHOCLO and Trueque using the public Hugging Face releases. Our runs are directly comparable at the dataset level, but not necessarily at the score level when the original authors use a different evaluation stack."

For CHOCLO specifically:

"We report our own metric suite and do not treat the published CHOCLO aggregate numbers as directly reproducible unless the same lexical, embedding, and LLM-judge pipeline is replicated."
