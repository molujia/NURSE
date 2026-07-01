# Release Audit Checklist

Use this checklist before submitting the artifact to HotCRP, publishing an anonymous review mirror, or creating a public release.

## Required

- [ ] The submitted URL or archive does not reveal author identity during double-anonymous review.
- [ ] `README.md` and `docs/ARTIFACT_EVALUATION.md` are present and readable.
- [ ] Installation commands work on a clean Python 3.10+ environment.
- [ ] The quick availability tests pass.
- [ ] `LICENSE` is present for the NURSE implementation.
- [ ] Dataset sources and original release terms are documented.
- [ ] Full benchmark replication commands are documented.
- [ ] OpenAI `gpt-5-nano` smoke commands are documented.
- [ ] SOTA full, E/P/H ablation, consistency, and aggregation entrypoints are documented.
- [ ] Output paths and expected summary files are documented.
- [ ] No API keys, private endpoints, local absolute paths, private logs, or human-participant identifiers are committed.
- [ ] Signed consent forms and identifiable human-study free text are excluded.
- [ ] Generated outputs, `tmp/`, virtual environments, caches, local datasets, and bundled benchmark metadata are excluded.

## Recommended

- [ ] Create an immutable post-review archive on Zenodo or Software Heritage.
- [ ] Add the archive DOI or persistent identifier to the paper and README after de-anonymization.
- [ ] Tag the exact commit used for the camera-ready artifact.
- [ ] Record dependency versions and LLM model names used for final experiments.
