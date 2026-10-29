import risk


def _analysis(severity="low", deviations=None, contract_text="The supplier delivers the goods.", evidence_id="payment-01"):
    return {
        "clause_id": "clause-001",
        "category": "Payment",
        "contract_text": contract_text,
        "evidence_id": evidence_id,
        "reference_clause": "The customer pays undisputed invoices within thirty days after receipt.",
        "deviations": [] if deviations is None else deviations,
        "explanation": "Wording differs from the reference.",
        "severity": severity,
        "warnings": ["Semantic similarity ranks retrieved passages only. It is not a legal risk rating."],
        "retrieved": [
            {
                "evidence_id": evidence_id,
                "text": "The customer pays undisputed invoices within thirty days after receipt.",
                "similarity": 0.8,
            }
        ],
    }


def test_low_severity_without_targeted_issue_is_safe():
    scored = risk.score_analysis(_analysis("low", ["The clause uses a different defined term."]))

    assert scored["label"] == "Safe"
    assert scored["state"] == "scored"
    assert scored["deciding_rule"] == "severity-low"
    assert scored["finding"] == "no-targeted-issue"
    assert scored["confirmed_violation"] is False


def test_medium_deviation_is_caution_not_risky():
    scored = risk.score_analysis(
        _analysis("medium", ["The invoice period is twenty days rather than thirty."])
    )

    assert scored["label"] == "Caution"
    assert scored["label"] != "Risky"
    assert scored["finding"] == "identified-issue"
    assert scored["confirmed_violation"] is False
    assert "payment-tighter-terms" in {item["id"] for item in scored["rules"]}


def test_high_severity_alone_is_not_risky():
    scored = risk.score_analysis(
        _analysis("high", ["The clause uses different defined terms throughout."])
    )

    assert scored["label"] == "Caution"
    assert scored["deciding_rule"] == "severity-high"


def test_major_issue_rules_are_risky():
    cases = {
        "termination-immediate-or-one-sided": "Only the supplier may terminate immediately without notice.",
        "liability-unlimited": "The customer accepts unlimited liability for every claim.",
        "indemnity-one-sided": "Only one party gives an indemnity for third-party claims.",
        "payment-restrictive": "Every invoice is due immediately and the customer cannot dispute it.",
        "confidentiality-without-limits": "Confidential information stays confidential forever and has no exceptions.",
        "renewal-narrow-window": "This agreement will automatically renew unless cancellation notice of 90 days is given.",
        "ip-broad-transfer": "The customer assigns all pre-existing intellectual property to the supplier.",
        "noncompete-restrictive": "The supplier shall not compete in any industry worldwide for five years.",
        "governing-law-unclear": "The supplier may change the governing law after signature.",
        "dispute-unbalanced": "The customer waives any right to injunctive relief.",
    }
    for rule_id, text in cases.items():
        scored = risk.score_analysis(_analysis("low", [text], contract_text=text))
        assert scored["label"] == "Risky", rule_id
        assert scored["deciding_rule"] == rule_id
        assert scored["confirmed_violation"] is False


def test_shorter_notice_is_caution_rather_than_risky():
    scored = risk.score_analysis(
        _analysis("low", ["The notice period is shorter than the reference."])
    )

    assert scored["label"] == "Caution"
    assert scored["deciding_rule"] == "termination-short-notice"


def test_missing_evidence_is_not_safe():
    result = _analysis("low")
    result["evidence_id"] = ""
    result["reference_clause"] = ""
    result["severity"] = ""
    result["warnings"].append("No relevant playbook clause was retrieved, so no comparison was made.")

    scored = risk.score_analysis(result)

    assert scored["label"] == "Review required"
    assert scored["state"] == "review-required"
    assert scored["label"] != "Safe"
    assert scored["warning"].startswith("Review required.")
    assert scored["deciding_rule"] == "missing-evidence"
    assert scored["confirmed_violation"] is False


def test_invalid_input_is_review_required():
    for value in (None, [], "not-an-analysis"):
        scored = risk.score_analysis(value)
        assert scored["label"] == "Review required"
        assert scored["state"] == "review-required"
        assert scored["deciding_rule"] == "invalid-input"

    incomplete = _analysis("low")
    incomplete["contract_text"] = "  "
    assert risk.score_analysis(incomplete)["deciding_rule"] == "incomplete-extraction"

    invalid = _analysis("critical")
    assert risk.score_analysis(invalid)["deciding_rule"] == "invalid-analysis"


def test_same_input_is_repeatable_and_does_not_call_a_model():
    sample = _analysis("low", ["The customer accepts uncapped liability for the services."])
    first = risk.score_analysis(sample)
    second = risk.score_analysis(sample)

    assert first == second
    assert first["label"] == "Risky"
    assert "openai" not in risk.__dict__
