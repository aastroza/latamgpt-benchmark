# Dataset citations and comparison notes

Last reviewed: 2026-04-22

This document records the official citation text exposed on the Hugging Face dataset cards and clarifies what is and is not directly comparable from this repository.

Sources:

- [Trueque dataset card](https://huggingface.co/datasets/latam-gpt/Trueque-Benchmark-beta-0.1)
- [CHOCLO dataset card](https://huggingface.co/datasets/latam-gpt/CHOCLO)

## Citation Policy For This Repository

If we publish benchmark results, reports, slides, blog posts, or model cards based on these datasets, we should cite both dataset cards and use the benchmark names consistently:

- Trueque: `Trueque: A human-reviewed collaborative benchmark for Latin American knowledge and culture`
- CHOCLO: `CHOCLO: Latin American Cultural Knowledge Benchmark`

## Official Citations

### Trueque

```bibtex
@software{Trueque_benchmark_beta_0.1,
  title={Trueque: A human-reviewed collaborative benchmark for Latin American knowledge and culture},
  author={Fuentes, Gonzalo and Arriagada, Alexandra and Henriquez, Clemente and García, M. Alexandra and LatamGPT Team},
  year={2026},
  url={https://huggingface.co/latam-gpt/Trueque-Benchmark-beta-0.1}
}
```

### CHOCLO

```bibtex
@dataset{choclo2026,
  title={CHOCLO: Latin American Cultural Knowledge Benchmark},
  author={Del Solar, Bianca and Carvallo, Andrés and Soto, Álvaro},
  year={2026},
  publisher={Hugging Face},
  note={Developed as part of doctoral research under the supervision of Álvaro Soto}
}
```

## Can We Compare Against The Published Results?

### Trueque

Not directly, at least not yet.

What the card says:

- this is still a beta release
- baseline results are still pending
- the evaluation setup is still being adjusted
- the current automated evaluation baseline is `Qwen2.5-Instruct (72B)`

Implication:

- There are no stable published headline numbers to compare against yet.
- The card does not expose enough detail to reproduce the exact current public evaluation pipeline from this repository alone.

Closest run from this repository today:

```bash
uv run latamgpt-benchmark submit --dataset trueque --model-suite benchmark-default
uv run latamgpt-benchmark collect --run-dir runs/<run-name> --wait
```

Why this is only approximate:

- our benchmark set is restricted to OpenAI and Doubleword
- our default judge is `openai:gpt-4.1-mini`, not the card's stated `Qwen2.5-Instruct (72B)`
- the card says the evaluation setup is still under revision

### CHOCLO

Partially, but not as an apples-to-apples score comparison.

What the card publishes:

- aggregate results over the benchmark
- model list including `GPT-4o Mini`, `GPT-3.5 Turbo`, `GPT-5 Mini`, `Mistral`, and `DeepSeek`
- a hybrid evaluation stack combining lexical similarity, embedding similarity, and LLM judging

Implication:

- We can align closely to the OpenAI lineage the card references.
- We still should not claim direct score comparability unless we reproduce the exact original metric stack and judge setup.

Closest run from this repository today for the OpenAI subset:

```bash
uv run latamgpt-benchmark submit --dataset choclo --model-suite openai-conservative
uv run latamgpt-benchmark collect --run-dir runs/<run-name> --wait
```

The practical OpenAI mapping used here is:

- `GPT-3.5 Turbo` -> `openai:gpt-4.1-mini`
- `GPT-5 Mini` -> `openai:gpt-5-mini`

The model we are no longer carrying in the default OpenAI suite is:

- `GPT-4o Mini`

Important caveat:

- This is a candidate-model comparison on the same benchmark questions.
- It is not a faithful reproduction of the published CHOCLO aggregate numbers.

Why not:

- our deterministic metrics are `normalized_exact_match`, `token_f1`, and `char_similarity`
- the card reports lexical, embedding, and LLM judge scores
- our judge prompt and judge model are not the original unpublished CHOCLO setup

## Recommended Wording For Future Reports

Use wording like this:

"We evaluate on CHOCLO and Trueque using the public Hugging Face releases. Our runs are directly comparable at the dataset level, but not necessarily at the score level when the original authors use a different evaluation stack."

For CHOCLO specifically:

"We report our own metric suite and do not treat the published CHOCLO aggregate numbers as directly reproducible unless the same lexical, embedding, and LLM-judge pipeline is replicated."
