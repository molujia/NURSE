# Report paths and root cause path accuracy

NURSE exposes core propagation-support paths, final report predictions, and
independently supplied reference paths as separate objects. PA evaluates the
report predictions against the references. A propagation support score, the
frozen core paths, or the G+ evidence graph alone is not the final predicted
path collection.

## Formula and matching

The offline calculator implements Root Cause Path Accuracy from
[mABC, Section 3.2, Equation 6](https://aclanthology.org/2024.findings-emnlp.232/):

```text
PA = (Pc - 0.2 * Pi) / Pt, for Pt > 0
```

`Pc` counts correctly identified paths, `Pi` counts incorrectly inferred paths,
and `Pt` is the number of actual reference paths. Negative values are retained;
the score is not clipped. The normalization, matching, and failure rules below
are NURSE's operational protocol.

A complete prediction is a directed service sequence from a proposed root to
a symptom, containing at least two known services with no repeated service.
Predictions use `root_to_symptom`. References explicitly marked
`symptom_to_root` are reversed before comparison. Instance and pod identities
must first be mapped to canonical services using a frozen incident-specific map.

Predictions and references are deduplicated independently by complete sequence
within each incident. Matching requires exact sequence equality, with no partial
credit for shared edges or endpoints. Every unique formally asserted prediction
is evaluated, including paths from multiple candidate roots, paths longer than
two hops, inferred missing edges, and inferred order within a strongly connected
component. Inference does not exempt a prediction from penalties.

Recognizable declarations with unknown services, repeated services, or a single
service remain incorrect predictions. Evidence-reference, edge-label, and
prose-binding errors are audited separately without removing recognizable paths.
An unrecoverable sequence makes the whole collection unscorable. Propagation
fragments belong outside the complete-path prediction collection.

## Report outputs

New typed and G+ response contracts require `predicted_paths` and
`path_abstention_reason`. Each path supplies its identity, complete
`service_sequence`, rationale, and ordered adjacent edges. Each edge declares
its source and target service, `basis` (`observed_relation` or `inferred`),
evidence references, and rationale. An observed relation does not by itself
prove fault propagation. Evidence audits use the material actually delivered
to the report request.

New G+ report owners save:

| File | Contents |
| --- | --- |
| `predicted-paths.json` | Raw and canonical predictions, edge declarations, audit issues, status, and source hashes. |
| `predicted-paths.md` | A readable view of the collection, identified as model hypotheses. |
| `report.md` | The accepted factual report with the same prediction view appended. |

Paths are exported before factual prose binding, so prose rejection does not
erase the independent prediction artifacts. Core scores and ranking remain
unchanged. Historical responses remain readable through their original contracts
without inventing empty prediction collections.

The export schema is `nurse.v2.report-path-predictions/1`. It binds the opaque
incident identity to core, context, request, native-response, and artifact hashes.
The calculator checks integrity and the complete raw-to-canonical projection.
Use the saved-report reader to verify the original response, request, context,
service map, and independently retained hashes. A self-contained hash alone
cannot authenticate source data if both data and hashes have been rewritten.

## Independent references

References use `nurse.v2.path-reference/1` and must identify the same incident
as the prediction export. Direction is `root_to_symptom` or `symptom_to_root`,
granularity is `service`, and status is `complete`, `incomplete`, `unavailable`,
or `not_applicable`. The reference supplies complete service sequences and
provenance. The calculator cannot certify independent annotation or completeness.
The [calculator module](../src/nurse_rcl/evaluation/path_accuracy.py) defines the
full input validation contract.

References must establish actual complete paths within a declared incident,
observation window, and symptom scope. Root labels, topology reachability,
anomaly summaries, and model support paths are not individually sufficient.
Preserve the evidence sources, service mapping, ambiguities, adjudication, and
basis for completeness. Use an incomplete or unavailable status when necessary.

Freeze references independently of evaluated reports. Reference paths, root
labels, and adjudication records must remain outside inference contexts, prompts,
tool feedback, and model-visible service aliases. Seal predictions before scoring
and do not revise references to fit predictions.

## Calculation and failure handling

Run from an installed checkout:

```bash
python -m nurse_rcl.evaluation.path_accuracy \
  --predictions PREDICTIONS_JSON \
  --reference REFERENCE_JSON
```

Replace the named placeholders with file paths. The CLI returns counts,
matched, incorrect, and missed sequences, calculation status, and source pins.
It performs no model calls.

| Condition | Handling |
| --- | --- |
| Complete nonempty reference and available predictions | Compute PA, retaining negative values. |
| Complete nonempty reference and explicit abstention with a reason | Score the declared empty prediction set. |
| Empty `available` collection or contradictory fields | Reject invalid input. |
| Missing, unavailable, or invalid native predictions | Return an unscorable status and null PA. |
| Incomplete or unavailable reference | Return null PA. |
| Zero reference paths or `not_applicable` | Exclude explicitly with a reason; PA is undefined. |
| Corrupt files, mismatched identity, or invalid hashes | Reject input and retain the failure in batch accounting. |

Missing output is not explicit abstention. Do not replace failures with numeric
scores or count predictions absent from an incomplete reference as incorrect.

`aggregate_path_accuracy(results)` reports incident-level `macro_pa` and pooled
`micro_pa`. Micro PA applies the same formula to summed counts, weighting
incidents by their number of reference paths. If an expected incident is
unscorable, the formal full-batch scores are null. `scorable_subset`, `coverage`,
and `status_counts` remain available for diagnostics. Explicit exclusions are
recorded separately. Empty batches have no mean, and duplicate identities are
rejected. A subset mean must not be presented as the complete cohort result.
