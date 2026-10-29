"""Deterministic risk labels for a validated clause comparison.

The labels Safe, Caution, and Risky are project review labels. They record
an identified wording issue. They are not a finding that a clause violates
the law, and these rules are not legally validated.

Severity from the comparison is only a starting point. A wording difference
becomes Risky only when a targeted rule below also matches. Retrieval
failure, incomplete extraction, and invalid analysis do not receive Safe.
"""

import re

import utils as Utils

SAFE = "Safe"
CAUTION = "Caution"
RISKY = "Risky"
REVIEW_REQUIRED = "Review required"
REVIEW_STATE = "review-required"

_RANK = {SAFE: 0, CAUTION: 1, RISKY: 2}

DISCLAIMER = (
    "This label follows project rules for wording differences. "
    "It is not a legally validated conclusion and it is not a finding that the clause violates the law."
)

_FAILURE_MARKERS = (
    "no comparison was made",
    "no comparison was accepted",
    "not retrieved",
    "not valid json",
    "was empty",
    "timed out",
    "rejected the api key",
    "request failed",
    "incomplete extraction",
    "extraction is incomplete",
)


class _Rule:
    def __init__(self, rule_id, effect, pattern, reason):
        self.id = rule_id
        self.effect = effect
        self.pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        self.reason = reason


def _rule(rule_id, effect, pattern, reason):
    return _Rule(rule_id, effect, pattern, reason)


# Targeted phrases. Each reason describes an identified wording issue only.
_CAUTION_RULES = (
    _rule(
        "termination-short-notice",
        CAUTION,
        r"notice period is shorter|cure period is shorter|less than (?:thirty|30) days['’]? notice",
        "A deviation describes a shorter notice or cure period. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "liability-narrow-cap",
        CAUTION,
        r"cap is (?:less|lower) than the fees|one[- ]sided exclusion of consequential damages",
        "A deviation describes a narrow liability cap or a one-sided damages exclusion. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "payment-tighter-terms",
        CAUTION,
        r"within ten days|late fee|late penalty|rather than thirty",
        "A deviation describes tighter payment timing or a late charge. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "confidentiality-longer-duty",
        CAUTION,
        r"confidentiality (?:period|duty) is longer|no compelled[- ]disclosure",
        "A deviation describes a longer confidentiality duty or a missing disclosure exception. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "renewal-needs-review",
        CAUTION,
        r"renews unless|price (?:may|can) increase on renewal",
        "A deviation describes renewal or a renewal price change that needs review. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "ip-ownership-difference",
        CAUTION,
        r"supplier keeps (?:the )?custom deliverables|no license to (?:the )?tools",
        "A deviation describes a narrower license or ownership of deliverables. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "noncompete-broad-solicit",
        CAUTION,
        r"non-solicit covers every employee|restriction lasts more than twelve months",
        "A deviation describes a broad employee non-solicit. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "governing-law-mismatch",
        CAUTION,
        r"venue (?:is|differs)|conflict-of-law rules are included",
        "A deviation describes a venue or conflict-of-law mismatch. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "dispute-process-difference",
        CAUTION,
        r"arbitration is required|waiting period before (?:a )?claim",
        "A deviation describes a different dispute process. This is an identified wording issue, not a confirmed legal violation.",
    ),
)

_RISKY_RULES = (
    _rule(
        "termination-immediate-or-one-sided",
        RISKY,
        r"terminat\w{0,12}.{0,48}(?:immediately|without (?:prior )?notice|no notice)|"
        r"without (?:prior )?notice.{0,48}terminat|"
        r"only (?:one party|the (?:supplier|customer|company|vendor)).{0,48}terminat|"
        r"no opportunity to cure",
        "The wording describes termination that is immediate, without notice, or available to only one party. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "liability-unlimited",
        RISKY,
        r"unlimited liability|liability is unlimited|uncapped liability|no cap on liability|without (?:any )?limit on liability",
        "The wording describes liability that is unlimited or has no cap. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "indemnity-one-sided",
        RISKY,
        r"one-sided indemn|only one party.{0,40}indemn|solely the .{0,24}indemn|"
        r"(?:customer|supplier|vendor|company) alone shall indemnif",
        "The wording describes an indemnity owed by only one party. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "payment-restrictive",
        RISKY,
        r"due immediately|payable immediately|due on receipt|payment before delivery|"
        r"cannot dispute|no right to dispute|payable in advance",
        "The wording describes payment that is immediate, required in advance, or cannot be disputed. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "confidentiality-without-limits",
        RISKY,
        r"confidential\w*.{0,48}(?:perpetual|indefinite|forever)|"
        r"(?:perpetual|indefinite|forever).{0,48}confidential|"
        r"confidential\w*.{0,48}no (?:exclusions|exceptions)|"
        r"no (?:exclusions|exceptions).{0,48}confidential",
        "The wording describes confidentiality with no stated limit or exception. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "renewal-narrow-window",
        RISKY,
        r"auto(?:matic(?:ally)?|-)?\s*renew\w*.{0,80}(?:90 days|60 days|sixty days|six months)|"
        r"cancel(?:lation)? notice of (?:at least )?(?:90|60) days|"
        r"no right to cancel",
        "The wording describes automatic renewal with a narrow or missing cancellation window. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "ip-broad-transfer",
        RISKY,
        r"assign(?:s|ment of)? all (?:pre-existing )?(?:intellectual property|ip\b)|"
        r"transfer(?:s)? all pre-existing|"
        r"ownership of all (?:intellectual property|pre-existing)",
        "The wording describes a broad transfer of intellectual property, including pre-existing material. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "noncompete-restrictive",
        RISKY,
        r"non-?compete.{0,70}(?:worldwide|any (?:business|industry)|indefinite|five years)|"
        r"shall not compete in any",
        "The wording describes a non-compete with a very broad scope or duration. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "governing-law-unclear",
        RISKY,
        r"governing law (?:is )?(?:not specified|unnamed|undetermined)|"
        r"(?:may|can) change the governing law|"
        r"governing law of (?:its|their) (?:own )?choice",
        "The wording leaves the governing law unnamed or lets one party change it. This is an identified wording issue, not a confirmed legal violation.",
    ),
    _rule(
        "dispute-unbalanced",
        RISKY,
        r"waive[sd]? (?:any )?(?:right to )?injunctive|"
        r"only (?:one party|the (?:company|supplier|customer)) may (?:bring|file|choose)|"
        r"mandatory arbitration.{0,50}(?:cannot|may not|no right to).{0,20}court",
        "The wording describes a dispute process that blocks court access or injunctive relief for one side. This is an identified wording issue, not a confirmed legal violation.",
    ),
)


def score_analysis(result):
    """Map one validated analysis record to a repeatable review label."""
    if not isinstance(result, dict):
        return _review("", "invalid-input", "The analysis input was not a result object, so it was not labeled Safe.")

    clause_id = str(result.get("clause_id") or "")
    problem = _review_problem(result)
    if problem is not None:
        rule_id, reason = problem
        return _review(clause_id, rule_id, reason)

    severity = str(result.get("severity") or "").strip().lower()
    baseline = _Rule(
        f"severity-{severity}",
        CAUTION if severity in {"medium", "high"} else SAFE,
        r"$^",
        _baseline_reason(severity),
    )
    chunks = _chunks(result)
    matched = [baseline]
    deciding = baseline
    for rule in _CAUTION_RULES + _RISKY_RULES:
        if any(rule.pattern.search(chunk) for chunk in chunks):
            matched.append(rule)
            if _RANK[rule.effect] >= _RANK[deciding.effect]:
                deciding = rule

    label = deciding.effect
    return {
        "clause_id": clause_id,
        "label": label,
        "state": "scored",
        "rules": [_public_rule(rule) for rule in matched],
        "deciding_rule": deciding.id,
        "warning": "",
        "finding": "identified-issue" if label != SAFE else "no-targeted-issue",
        "confirmed_violation": False,
        "disclaimer": DISCLAIMER,
    }


def score_analyses(results):
    if not results:
        return []
    return [score_analysis(result) for result in results]


def _baseline_reason(severity):
    if severity == "low":
        return (
            "The comparison severity is low and no targeted issue rule matched. "
            "A difference from the reference, by itself, is not labeled Risky."
        )
    if severity == "medium":
        return (
            "The comparison severity is medium. The wording difference is labeled Caution "
            "unless a targeted Risky rule also matches. This is not a confirmed legal violation."
        )
    return (
        "The comparison severity is high. A large wording difference is labeled Caution, "
        "not Risky, unless a targeted Risky rule also matches. This is not a confirmed legal violation."
    )


def _review_problem(result):
    contract_text = result.get("contract_text")
    if not isinstance(contract_text, str) or not contract_text.strip():
        return (
            "incomplete-extraction",
            "The contract text is missing or empty, so the clause was not labeled Safe.",
        )
    if result.get("extraction_complete") is False:
        return (
            "incomplete-extraction",
            "Extraction was marked incomplete, so the clause was not labeled Safe.",
        )
    evidence_id = str(result.get("evidence_id") or "").strip()
    reference = result.get("reference_clause")
    severity = str(result.get("severity") or "").strip().lower()
    deviations = result.get("deviations")
    if not evidence_id or not isinstance(reference, str) or not reference.strip():
        return (
            "missing-evidence",
            "No retrieved reference was available, so the clause was not labeled Safe.",
        )
    if severity not in Utils.SEVERITY_LEVELS or not isinstance(deviations, list):
        return (
            "invalid-analysis",
            "The comparison severity or deviations were not valid, so the clause was not labeled Safe.",
        )
    retrieved = result.get("retrieved")
    if isinstance(retrieved, list) and retrieved:
        retrieved_ids = {str(hit.get("evidence_id") or "") for hit in retrieved if isinstance(hit, dict)}
        if evidence_id not in retrieved_ids:
            return (
                "invalid-analysis",
                "The evidence id was not in the retrieved passages, so the clause was not labeled Safe.",
            )
    warnings = result.get("warnings")
    if isinstance(warnings, list):
        combined = " ".join(str(item) for item in warnings).lower()
        if any(marker in combined for marker in _FAILURE_MARKERS):
            return (
                "invalid-analysis",
                "The comparison reported a failure, so the clause was not labeled Safe.",
            )
    return None


def _chunks(result):
    chunks = [result.get("contract_text") or ""]
    for item in result.get("deviations") or []:
        if isinstance(item, str):
            chunks.append(item)
    return chunks


def _review(clause_id, rule_id, reason):
    warning = "Review required. " + reason
    return {
        "clause_id": clause_id,
        "label": REVIEW_REQUIRED,
        "state": REVIEW_STATE,
        "rules": [{"id": rule_id, "effect": REVIEW_REQUIRED, "reason": reason}],
        "deciding_rule": rule_id,
        "warning": warning,
        "finding": "not-scored",
        "confirmed_violation": False,
        "disclaimer": DISCLAIMER,
    }


def _public_rule(rule):
    return {"id": rule.id, "effect": rule.effect, "reason": rule.reason}
