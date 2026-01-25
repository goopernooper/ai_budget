from app.services.ingestion import parse_transactions_csv


def test_parse_transactions_csv_basic():
    csv_data = (
        "date,description,amount,category\n"
        "2024-01-01,Uber Ride,-15.75,Transport\n"
        "2024-01-02,Salary Deposit,2500,Income\n"
    ).encode("utf-8")

    records = parse_transactions_csv(csv_data)
    assert len(records) == 2
    assert records[0]["amount"] == -15.75
    assert records[1]["amount"] == 2500.0
    assert records[0]["category"] == "Transport"


def test_parse_transactions_csv_missing_columns():
    csv_data = "date,amount\n2024-01-01,-20\n".encode("utf-8")
    try:
        parse_transactions_csv(csv_data)
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing columns")
