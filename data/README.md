# data/

Local cache of the ScienceQA image-only split (`derek-thomas/ScienceQA` on HuggingFace), restricted to questions that have an associated image.

```
scienceqa/
  train/           train split images
  train.json       train annotations
  validation/      val split images
  validation.json  val annotations
  test/            test split images
  test.json        test annotations
```

Split sizes after image-only filtering: **6,218 train / 2,097 val / 2,017 test**. Single-expert ablations are evaluated on a 500-sample subset of test; the MoC training loop validates on a 200-sample subset of validation every 750 steps and reports its final eval on a 500-sample test subset; the full 2,017-sample test split is used for the subject-level breakdown and the 100-error failure audit.
