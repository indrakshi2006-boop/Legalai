import json

import openai

import analyze


PAYMENT_HITS = [
    {
        "evidence_id": "payment-01",
        "text": "The customer pays undisputed invoices within thirty days after receipt.",
        "category": "Payment",
        "title": "Invoice timing",
        "distance": 0.2,
        "similarity": 0.8,
    },
    {
        "evidence_id": "payment-02",
        "text": "A late undisputed amount accrues interest at one percent per month.",
        "category": "Payment",
        "title": "Late amounts",
        "distance": 0.3,
        "similarity": 0.7,
    },
    {
        "evidence_id": "payment-03",
        "text": "The customer may withhold the disputed portion of an invoice.",
        "category": "Payment",
        "title": "Disputed invoices",
        "distance": 0.4,
        "similarity": 0.6,
    },
    {
        "evidence_id": "termination-01",
        "text": "Either party may end this agreement for convenience on thirty days' notice.",
        "category": "Termination",
        "title": "Termination for convenience",
        "distance": 0.55,
        "similarity": 0.45,
    },
]


def _retriever_factory(hits):
    calls = []

    def retriever(text, category=None, k=3):
        calls.append({"category": category, "k": k, "text": text})
        chosen = hits if category is None else [hit for hit in hits if hit["category"] == category]
        return chosen[:k]

    retriever.calls = calls
    return retriever


def _clause():
    return {
        "id": "clause-001",
        "text": "2. The customer pays each invoice within ten days of receipt.",
        "sequence": 1,
    }


def test_payment_clause_filters_by_category_and_keeps_retrieved_text():
    retriever = _retriever_factory(PAYMENT_HITS)

    def comparer(contract_text, hits):
        assert "ignore previous instructions" not in contract_text
        assert [hit["evidence_id"] for hit in hits] == ["payment-01", "payment-02", "payment-03"]
        return {
            "evidence_id": "payment-01",
            "reference_clause": "This invented reference must not be stored.",
            "deviations": ["Payment is due in ten days rather than thirty."],
            "explanation": "The contract shortens the invoice period.",
            "severity": "medium",
        }

    result = analyze.analyze_clause(_clause(), 0, retriever, comparer)

    assert result["clause_id"] == "clause-001"
    assert result["category"] == "Payment"
    assert result["contract_text"] == _clause()["text"]
    assert result["evidence_id"] == "payment-01"
    assert result["reference_clause"] == PAYMENT_HITS[0]["text"]
    assert result["reference_clause"] != "This invented reference must not be stored."
    assert result["severity"] == "medium"
    assert result["deviations"] == ["Payment is due in ten days rather than thirty."]
    assert [hit["evidence_id"] for hit in result["retrieved"]] == [
        "payment-01",
        "payment-02",
        "payment-03",
    ]
    assert result["retrieved"][0]["similarity"] == 0.8
    assert retriever.calls[0]["category"] is None
    analyze.validate_analysis(result)


def test_uncertain_category_searches_broadly():
    mixed = [
        dict(PAYMENT_HITS[3], similarity=0.5, distance=0.5),
        dict(PAYMENT_HITS[0], similarity=0.48, distance=0.52),
    ]
    retriever = _retriever_factory(mixed)
    seen = {}

    def comparer(contract_text, hits):
        seen["ids"] = [hit["evidence_id"] for hit in hits]
        return {
            "evidence_id": hits[0]["evidence_id"],
            "deviations": ["The notice period is not stated."],
            "explanation": "The clause does not track the nearest reference.",
            "severity": "low",
        }

    result = analyze.analyze_clause(
        {"id": "clause-002", "text": "The parties will cooperate on scheduling.", "sequence": 2},
        1,
        retriever,
        comparer,
    )

    assert result["category"] == ""
    assert seen["ids"] == ["termination-01", "payment-01"]
    assert all(call["category"] is None for call in retriever.calls)
    assert any("uncertain" in warning for warning in result["warnings"])
    analyze.validate_analysis(result)


def test_invented_evidence_id_is_rejected():
    retriever = _retriever_factory(PAYMENT_HITS)

    def comparer(contract_text, hits):
        return {
            "evidence_id": "made-up-99",
            "reference_clause": "A reference the model invented.",
            "deviations": ["Invented deviation."],
            "explanation": "Invented explanation.",
            "severity": "high",
        }

    result = analyze.analyze_clause(_clause(), 0, retriever, comparer)

    assert result["evidence_id"] == ""
    assert result["reference_clause"] == ""
    assert result["deviations"] == []
    assert result["explanation"] == ""
    assert result["severity"] == ""
    assert result["retrieved"][0]["evidence_id"] == "payment-01"
    assert any("not retrieved" in warning for warning in result["warnings"])
    analyze.validate_analysis(result)


def test_low_similarity_does_not_call_the_model():
    weak = [dict(PAYMENT_HITS[0], similarity=0.1, distance=0.9)]
    retriever = _retriever_factory(weak)
    called = {"value": False}

    def comparer(contract_text, hits):
        called["value"] = True
        return {}

    result = analyze.analyze_clause(_clause(), 0, retriever, comparer)

    assert called["value"] is False
    assert result["evidence_id"] == ""
    assert any("No relevant" in warning for warning in result["warnings"])
    analyze.validate_analysis(result)


def test_model_failure_preserves_contract_text_and_passages():
    retriever = _retriever_factory(PAYMENT_HITS)

    def comparer(contract_text, hits):
        raise openai.APITimeoutError(request=None)

    result = analyze.analyze_clause(_clause(), 0, retriever, comparer)

    assert result["contract_text"] == _clause()["text"]
    assert result["evidence_id"] == ""
    assert result["retrieved"][0]["text"] == PAYMENT_HITS[0]["text"]
    assert any("timed out" in warning for warning in result["warnings"])
    analyze.validate_analysis(result)


def test_invalid_model_json_is_not_accepted():
    retriever = _retriever_factory(PAYMENT_HITS)

    def comparer(contract_text, hits):
        return "not json"

    result = analyze.analyze_clause(_clause(), 0, retriever, comparer)

    assert result["evidence_id"] == ""
    assert any("not valid JSON" in warning for warning in result["warnings"])
    analyze.validate_analysis(result)


def test_schema_rejects_unretrieved_evidence():
    bad = {
        "clause_id": "clause-001",
        "category": "Payment",
        "contract_text": "Pay now.",
        "evidence_id": "not-retrieved",
        "reference_clause": "Some other text.",
        "deviations": ["A deviation."],
        "explanation": "Because.",
        "severity": "high",
        "warnings": ["Semantic similarity ranks retrieved passages only. It is not a legal risk rating."],
        "retrieved": [
            {
                "evidence_id": "payment-01",
                "text": "Pay within thirty days.",
                "category": "Payment",
                "title": "Invoice timing",
                "distance": 0.2,
                "similarity": 0.8,
            }
        ],
    }
    try:
        analyze.validate_analysis(bad)
    except ValueError as exc:
        assert "not retrieved" in str(exc)
    else:
        raise AssertionError("expected the invented evidence id to fail validation")


def test_similarity_helper_is_not_used_as_severity():
    assert analyze.similarity_from_distance(0.2) == 0.8
    assert set(analyze.SEVERITY_LEVELS) == {"low", "medium", "high"}


def test_model_payload_parser_accepts_json_object():
    parsed = analyze._parse_model_json(
        "```json\n"
        + json.dumps(
            {
                "evidence_id": "payment-01",
                "deviations": ["Due on receipt."],
                "explanation": "Shorter period.",
                "severity": "low",
            }
        )
        + "\n```"
    )
    assert parsed["evidence_id"] == "payment-01"
    assert parsed["severity"] == "low"
