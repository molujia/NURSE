"""Versioned instructions for existing report guards; no new evidence authority."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nurse_rcl.artifacts import canonical_json_bytes, sha256_hex
from nurse_rcl.v2.exchange_json import ExchangeLimits, decode_json_object, encode_json
from nurse_rcl.v2.live import report_prompt as legacy

GUIDED_PROMPT_VERSION = 'nurse-v2-live-compact-report-prompt/2'
GUIDED_INSTRUCTIONS = legacy.INSTRUCTIONS + '''
Before composing, decode each cited row and inspect its own kind, targets, intervals,
values, scope_basis and allowed_claim_strength. Do not guess a row's values from a
nearby row or its alias number. Atom numbers are NOT fact aliases.

The following are the existing mechanical requirements, not optional style advice:
1. Ranking is inference, never describe_observation. Use explain_priority with
   exactly one target and its own ranking fact; interval must be null. Do not cite
   another candidate's ranking in a single-target priority claim.
2. compare_candidates requires exactly two targets, in the exact ordered pair of
   a cited competition fact. Cite that competition fact as the comparison basis;
   two ranking facts alone are insufficient. interval must be null.
3. describe_observation requires exactly one target, a non-null interval copied
   exactly from the cited observation/predicate, scope_basis=bound_observation and
   allowed_claim_strength=frozen_observation. Every non-context cited fact must be
   a matching observation/predicate for that same target and interval. A frozen
   score, hypothesis or declared_hypothesis predicate is NOT an observed fact.
4. synthesize_evidence requires exactly one target and at least two cited diagnostic
   facts about it (observation, predicate, hypothesis, ranking, candidate_evidence).
   Distinct aliases, especially grouped aliases, do not establish independence.
5. explain_uncertainty permits zero targets (global) or exactly one target. Cite
   actual uncertainty: a limit, unknown/adverse predicate, missing/degraded record,
   positive unknown/refuting/missing/conflict count, limited/mixed/insufficient
   evidence, actual tie, or explicit degradation. tie_break_is_evidence=false alone
   is not an uncertainty basis and does not mean that a tie occurred.
6. propose_verification requires exactly one target and cited observation/predicate/
   hypothesis evidence for it. Its nonempty precondition_refs must cite only those
   evidence kinds, not rankings, summaries or global limits. Use read_only_diagnostic_check,
   status=proposed and true/false/unknown outcomes. Never propose to verify a rank
   as if it were a physical observation. A verification card alone is not enough.
7. conditional_mitigation requires exactly one target and a cited mitigation fact
   for that target with the same action_class. Cite every individual true predicate
   required by that specific mitigation in precondition_refs. Do not substitute
   grouped predicates, ranking facts or global uncertainty for prerequisites.
   If no matching mitigation and complete true prerequisites are visible, omit it.
   Recommendations are optional, not a quota; retain proposed status and all safeguards.

For every claim, all cited fact targets must lie within its declared target(s).
The only shared-target exception is a declared_hypothesis predicate covering the
single claim target. Untargeted facts can accompany targeted claims only if they
are uncertainty/limit facts. Other operations permit null interval; if specifying
one, it must exactly match every cited temporal fact, not a union or invented range.
Mandatory frozen candidate_evidence and global uncertainty/limit facts are context:
they can accompany an otherwise legal claim without serving as an incompatible
primary fact. Global adverse context is also attached deterministically; do not
force it into incompatible claims or omit uncertainty from the explanation.

Each assertion must copy a name/value from the values pairs of the same fact_ref.
Copy the decoded scalar's exact JSON type: score strings stay strings, counts stay
integers. Only values-pair names are assertion fields; kind, scope_basis and status_codes
are not values unless separately present there. A value from fB must never be asserted
against fA. Do not invent has_unknown_evidence on a candidate summary that only supplies
unknown_predicate_count. Do not turn a derived difference into a source assertion.

Primary fact kinds: explain_priority may use diagnostic facts and uncertainty;
compare_candidates may additionally use competition; synthesize_evidence uses
diagnostic facts; explain_uncertainty may additionally use competition/uncertainty/limit;
verification uses observation/predicate/hypothesis/verification; mitigation uses
observation/predicate/hypothesis/mitigation. All operation-specific bases still apply.

Prefer 3 to 6 concise, useful claims within 4096 output tokens: explain the highest
priority using available mechanism/evidence, a supported comparison if useful, and
uncertainty/limitations; add grounded read-only verification only when helpful.
Avoid duplicating a separate numerical recital for every rank. This preference
does not remove input facts or change the schema's ceilings. Select and synthesize
evidence freely within the rules; do not merely copy a stock template. Before output,
check each operation, target set, interval, assertion owner/type and recommendation
basis against the exact decoded rows. Never rewrite the evidence to make a claim pass.
'''


@dataclass(frozen=True, slots=True)
class LiveGuidedPrompt:
    payload_json: bytes

    def _data(self) -> dict[str, Any]:
        if type(self) is not LiveGuidedPrompt:
            raise ValueError('exact guided prompt required')
        raw = object.__getattribute__(self, 'payload_json')
        limits = ExchangeLimits(max_bytes=524288, max_nodes=100000)
        data = decode_json_object(raw, limits)
        if (set(data) != {'prompt_version', 'instructions', 'output_schema_json', 'data'}
                or data['prompt_version'] != GUIDED_PROMPT_VERSION
                or data['instructions'] != GUIDED_INSTRUCTIONS
                or type(data['data']) is not dict
                or data['data'].get('prompt_version') != GUIDED_PROMPT_VERSION
                or canonical_json_bytes(data) != raw):
            raise ValueError('guided prompt shape differs')
        base = dict(data, prompt_version=legacy.VERSION, instructions=legacy.INSTRUCTIONS,
                    data=dict(data['data'], prompt_version=legacy.VERSION))
        # All unchanged schema, compact wire and exact-meaning checks still apply.
        legacy.LiveCompactPrompt(encode_json(base, limits))._data()
        return data

    @property
    def sha256(self) -> str:
        self._data()
        return sha256_hex(self.payload_json)

    def messages(self) -> list[dict[str, str]]:
        data = self._data()
        return [dict(role='system', content=data['instructions']+'\nOutput schema:\n'+data['output_schema_json']),
                dict(role='user', content=canonical_json_bytes(data['data']).decode())]


def guide_prompt(prompt: legacy.LiveCompactPrompt) -> LiveGuidedPrompt:
    data = prompt._data()
    guided = dict(data, prompt_version=GUIDED_PROMPT_VERSION, instructions=GUIDED_INSTRUCTIONS,
                  data=dict(data['data'], prompt_version=GUIDED_PROMPT_VERSION))
    result = LiveGuidedPrompt(encode_json(guided, ExchangeLimits(max_bytes=524288, max_nodes=100000)))
    result._data()
    return result
