import embed
from segment import MAX_CLAUSE_CHARS, segment_contract


def _joined(clauses):
    return "".join(clause["text"] for clause in clauses)


def test_numbered_clauses():
    text = (
        "1. The supplier delivers the goods to the location named in the order.\n"
        "2. The customer pays undisputed invoices within thirty days of receipt.\n"
        "3. Either party may end the agreement by giving thirty days' written notice.\n"
    )
    clauses = segment_contract(text)

    assert [clause["sequence"] for clause in clauses] == [1, 2, 3]
    assert [clause["id"] for clause in clauses] == ["clause-001", "clause-002", "clause-003"]
    assert [clause["heading"][:2] for clause in clauses] == ["1.", "2.", "3."]
    assert _joined(clauses) == text


def test_nested_numbering():
    text = (
        "1. This agreement covers the services described in the order form.\n"
        "1.1 The supplier sends an invoice after each month of work is complete.\n"
        "1.1.1 The customer may withhold a disputed portion when it explains the dispute.\n"
        "2. The initial term is twelve months from the start date written above.\n"
    )
    clauses = segment_contract(text)

    assert len(clauses) == 4
    assert [clause["heading"].split()[0] for clause in clauses] == [
        "1.",
        "1.1",
        "1.1.1",
        "2.",
    ]
    assert _joined(clauses) == text
    assert clauses[0]["sequence"] < clauses[2]["sequence"]


def test_heading_based_clauses():
    text = (
        "Article 1\n"
        "Either party may end the agreement by giving thirty days' written notice.\n"
        "\n"
        "Section 2\n"
        "The customer pays undisputed invoices within thirty days of receipt.\n"
        "\n"
        "Clause 3\n"
        "Each party keeps intellectual property it owned before this agreement.\n"
        "\n"
        "TERMINATION FOR CONVENIENCE\n"
        "Ending the agreement for convenience does not create a termination penalty.\n"
    )
    clauses = segment_contract(text)
    headings = [clause["heading"] for clause in clauses]

    assert headings[0].startswith("Article 1")
    assert any(heading.startswith("Section 2") for heading in headings)
    assert any(heading.startswith("Clause 3") for heading in headings)
    assert any(heading.startswith("TERMINATION FOR CONVENIENCE") for heading in headings)
    assert _joined(clauses) == text
    assert all(clause["heading"] in clause["text"] for clause in clauses)


def test_unnumbered_paragraphs():
    text = (
        "The supplier delivers the goods to the location named in the order.\n"
        "\n"
        "The customer pays undisputed invoices within thirty days of receipt.\n"
        "\n"
        "Either party may end the agreement by giving thirty days' written notice.\n"
    )
    clauses = segment_contract(text)

    assert len(clauses) == 3
    assert clauses[0]["heading"].startswith("The supplier delivers")
    assert clauses[1]["text"].strip().startswith("The customer pays")
    assert _joined(clauses) == text


def test_empty_input():
    assert segment_contract("") == []
    assert segment_contract(" \n\t\n ") == []
    assert segment_contract(None) == []
    assert segment_contract(123) == []


def test_long_clauses_keep_original_wording():
    sentence = "The supplier performs the described services with reasonable care. "
    body = sentence * ((MAX_CLAUSE_CHARS // len(sentence)) + 8)
    text = (
        f"1. {body}\n"
        "2. The customer pays undisputed invoices within thirty days of receipt.\n"
    )
    clauses = segment_contract(text)

    assert len(clauses) > 2
    assert clauses[-1]["heading"].startswith("2.")
    assert _joined(clauses) == text
    assert all(len(clause["text"]) <= MAX_CLAUSE_CHARS + len(sentence) for clause in clauses[:-1])


def test_original_text_is_preserved():
    text = (
        "Preamble. This order is the whole deal between the parties.\n"
        "1. The supplier delivers the goods to the named location.\n"
        "1.1 Delivery occurs on a business day agreed by the parties.\n"
        "Article 2\n"
        "The customer pays undisputed invoices within thirty days.\n"
    )
    assert _joined(segment_contract(text)) == text


def test_short_fragment_merges_with_neighbour():
    text = (
        "1. Fees\n"
        "1.1 The customer pays undisputed invoices within thirty days of receipt.\n"
    )
    clauses = segment_contract(text)

    assert len(clauses) == 1
    assert clauses[0]["text"] == text
    assert clauses[0]["heading"].startswith("1. Fees")


def test_contract_ingestion_uses_clause_segmentation():
    text = (
        "1. The supplier delivers the goods to the location named in the order.\n"
        "2. The customer pays undisputed invoices within thirty days of receipt.\n"
    )
    assert embed.contract_clauses(text) == segment_contract(text)


# --- Preamble and other document structures -------------------------------

import segment as _segment


def _kinds(text):
    return [(clause["heading"][:24], clause["kind"]) for clause in segment_contract(text)]


def test_banner_before_numbered_clauses_is_preamble():
    text = (
        "FICTIONAL SAMPLE. This document is not a real contract and is used for testing only.\n"
        "1. Services. The supplier prepares the monthly newsletter described in the order.\n"
        "2. Payment. The customer pays each undisputed invoice within thirty days.\n"
    )
    clauses = segment_contract(text)
    assert [clause["kind"] for clause in clauses] == ["preamble", "clause", "clause"]
    assert len(_segment.reviewable(clauses)) == 2
    assert _joined(clauses) == text


def test_title_and_parties_block_is_preamble():
    text = (
        "SERVICES AGREEMENT\n"
        "This Agreement is made between Alpha Ltd and Beta LLC on the date below.\n"
        "1. Definitions\n"
        "1.1 \"Services\" means the work described in Schedule 1 of this agreement.\n"
        "1.2 \"Fees\" means the amounts listed in Schedule 2 of this agreement.\n"
        "2. Payment\n"
        "2.1 The Customer pays each undisputed invoice within thirty days of receipt.\n"
    )
    clauses = segment_contract(text)
    assert clauses[0]["kind"] == "preamble"
    assert clauses[0]["heading"] == "SERVICES AGREEMENT"
    assert all(clause["kind"] == "clause" for clause in clauses[1:])
    assert clauses[1]["heading"].startswith("1. Definitions")
    assert "1.1" in clauses[1]["text"]
    assert _joined(clauses) == text


def test_all_caps_headings_with_a_title():
    text = (
        "MASTER SERVICES AGREEMENT\n\n"
        "This agreement is entered into by the parties named below on the date of signature.\n\n"
        "PAYMENT\nThe customer pays undisputed invoices within thirty days of receipt.\n\n"
        "TERMINATION\nEither party may end the agreement by giving thirty days' written notice.\n"
    )
    kinds = _kinds(text)
    assert kinds[0] == ("MASTER SERVICES AGREEMEN", "preamble")
    assert [kind for _, kind in kinds[1:]] == ["clause", "clause"]


def test_all_caps_headings_without_a_title_are_all_clauses():
    text = (
        "PAYMENT\nThe customer pays undisputed invoices within thirty days of receipt.\n\n"
        "TERMINATION\nEither party may end the agreement by giving thirty days' written notice.\n"
    )
    assert [kind for _, kind in _kinds(text)] == ["clause", "clause"]


def test_title_does_not_absorb_the_first_article():
    text = (
        "MASTER AGREEMENT\n"
        "ARTICLE 1 - PAYMENT\nThe customer pays undisputed invoices within thirty days of receipt.\n"
        "ARTICLE 2 - TERMINATION\nEither party may end the agreement by giving thirty days' written notice.\n"
    )
    kinds = _kinds(text)
    assert kinds == [
        ("MASTER AGREEMENT", "preamble"),
        ("ARTICLE 1 - PAYMENT", "clause"),
        ("ARTICLE 2 - TERMINATION", "clause"),
    ]


def test_lettered_sub_clauses_stay_with_their_parent():
    text = (
        "1. Termination. Either party may terminate this agreement if:\n"
        "(a) the other party commits a material breach and does not cure it within fifteen days;\n"
        "(b) the other party becomes insolvent; or\n"
        "(c) the other party stops trading.\n"
        "2. Liability. Each party's total liability is capped at the fees paid in the prior year.\n"
    )
    clauses = segment_contract(text)
    assert len(clauses) == 2
    assert "(c) the other party stops trading." in clauses[0]["text"]
    assert all(clause["kind"] == "clause" for clause in clauses)


def test_table_cells_with_numbers_are_not_clause_starts():
    text = (
        "1. Fees. The customer pays the fees in the table below.\n"
        "Item\nAmount\nDue\nSetup\n500 USD\n10 days\nSupport\n200 USD\n30 days\n1.5 hours\n"
        "2. Termination. Either party may end this agreement on thirty days' written notice.\n"
    )
    clauses = segment_contract(text)
    assert [clause["heading"][:14] for clause in clauses] == ["1. Fees. The c", "2. Termination"]
    assert "500 USD" in clauses[0]["text"] and "1.5 hours" in clauses[0]["text"]
    assert _joined(clauses) == text


def test_a_single_block_is_never_all_preamble():
    clauses = segment_contract("SERVICES AGREEMENT\n")
    assert [clause["kind"] for clause in clauses] == ["clause"]


def test_unnumbered_document_has_no_preamble_unless_titled():
    text = (
        "The supplier delivers the goods to the location named in the order.\n\n"
        "The customer pays undisputed invoices within thirty days of receipt.\n"
    )
    assert all(clause["kind"] == "clause" for clause in segment_contract(text))


def test_record_start_offsets_point_into_the_text():
    text = "TITLE AGREEMENT\n1. The supplier delivers goods.\n2. The customer pays invoices on time.\n"
    for clause in segment_contract(text):
        assert text[clause["start"] : clause["start"] + len(clause["text"])] == clause["text"]
