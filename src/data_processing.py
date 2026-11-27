"""Prepare point-in-time-safe Kestrel returns datasets without training a model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ORDER_COLUMNS = (
    "order_id",
    "order_placed_at",
    "customer_id",
    "sku",
    "sales_channel",
    "payment_mode",
    "discount_pct",
    "qty",
    "order_value_inr",
    "promised_delivery_days",
    "delivery_pincode",
    "is_gift",
    "customer_prior_orders",
    "customer_prior_returns",
    "delivery_note",
    "last_service_event_type",
    "pickup_scheduled_at",
    "source",
)
TARGET_COLUMN = "returned"
REFERENCE_COLUMNS = {
    "customers": {"customer_id", "city", "state", "signup_date", "shield_member"},
    "products": {
        "sku",
        "family",
        "model_name",
        "list_price_inr",
        "warranty_months",
        "launch_date",
    },
}
FEATURE_COLUMNS = (
    "sales_channel",
    "payment_mode",
    "discount_pct",
    "qty",
    "promised_delivery_days",
    "delivery_pincode",
    "has_default_pincode",
    "is_gift",
    "order_hour",
    "order_day_of_week",
    "order_day",
    "order_month",
    "is_weekend",
    "sku",
    "product_family",
    "product_model_name",
    "product_age_days",
)
OUTPUT_METADATA_COLUMNS = ("order_id", "order_placed_at")
MATURITY_STATUS = "unverified"


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _find_one(data_dir: Path, suffix: str) -> Path:
    matches = sorted(data_dir.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*{suffix}' file in {data_dir}; found {len(matches)}."
        )
    return matches[0]


def _parse_order_times(frame: pd.DataFrame, name: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(frame["order_placed_at"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains invalid order_placed_at values.") from exc
    if parsed.isna().any():
        raise ValueError(f"{name} contains missing order_placed_at values.")
    if getattr(parsed.dt, "tz", None) is not None:
        raise ValueError(
            f"{name} order_placed_at must be timezone-naive IST as documented."
        )
    return parsed


def canonicalize_train(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Keep CRM rows only for exact CRM/partner-feed re-import pairs."""
    _require_columns(train, set(ORDER_COLUMNS) | {TARGET_COLUMN}, "train")
    if train["order_id"].isna().any():
        raise ValueError("train contains a missing order_id.")
    if not set(train[TARGET_COLUMN].dropna().unique()).issubset({0, 1}):
        raise ValueError("returned must contain only binary 0/1 values.")
    if train[TARGET_COLUMN].isna().any():
        raise ValueError("train contains a missing returned target.")
    if not set(train["source"].dropna().unique()).issubset({"crm", "partner_feed"}):
        raise ValueError("train.source contains an undocumented value.")
    if train["source"].isna().any():
        raise ValueError("train contains a missing source value.")

    equality_columns = [column for column in train.columns if column != "source"]
    kept_indices: list[Any] = []
    discarded_rows: list[dict[str, Any]] = []
    compared_groups = 0
    conflict_groups: dict[str, list[str]] = {}

    for order_id, group in train.groupby("order_id", sort=False, dropna=False):
        if len(group) == 1:
            kept_indices.append(group.index[0])
            continue

        compared_groups += 1
        conflicts = [
            column
            for column in equality_columns
            if group[column].nunique(dropna=False) > 1
        ]
        if conflicts:
            conflict_groups[str(order_id)] = conflicts
            continue

        sources = group["source"].tolist()
        if len(group) != 2 or set(sources) != {"crm", "partner_feed"}:
            conflict_groups[str(order_id)] = ["source/pair cardinality"]
            continue

        canonical = group.loc[group["source"].eq("crm")].iloc[0]
        discarded = group.loc[group["source"].eq("partner_feed")].iloc[0]
        kept_indices.append(canonical.name)

        discarded_record = discarded.to_dict()
        discarded_record.update(
            {
                "canonical_source": "crm",
                "discard_reason": "exact_match_partner_feed_reimport",
                "conflicting_fields": "",
            }
        )
        discarded_rows.append(discarded_record)

    if conflict_groups:
        examples = list(conflict_groups.items())[:10]
        raise ValueError(
            "Duplicate order IDs contain conflicting fields or unexpected source "
            f"pairings; no rows were discarded. Examples: {examples}"
        )

    canonical = train.loc[kept_indices].copy()
    if canonical["order_id"].duplicated().any():
        raise AssertionError("Canonicalization did not produce unique order IDs.")

    audit_columns = list(train.columns) + [
        "canonical_source",
        "discard_reason",
        "conflicting_fields",
    ]
    audit = pd.DataFrame(discarded_rows, columns=audit_columns)
    if len(audit) != compared_groups:
        raise AssertionError(
            "Every duplicate group must have exactly one audited discarded row."
        )

    stats = {
        "input_rows": len(train),
        "canonical_rows": len(canonical),
        "duplicate_groups": compared_groups,
        "discarded_rows": len(audit),
        "conflicting_groups": 0,
    }
    return canonical, audit, stats


def build_return_request_label_audit(canonical_train: pd.DataFrame) -> pd.DataFrame:
    """Create one-sided request labels without inventing negative outcomes."""
    required = {
        "order_id",
        TARGET_COLUMN,
        "pickup_scheduled_at",
        "last_service_event_type",
    }
    _require_columns(canonical_train, required, "canonical_train")
    if canonical_train["order_id"].isna().any() or canonical_train[
        "order_id"
    ].duplicated().any():
        raise ValueError("canonical_train order_id values must be present and unique.")
    if canonical_train[TARGET_COLUMN].isna().any() or not set(
        canonical_train[TARGET_COLUMN].unique()
    ).issubset({0, 1}):
        raise ValueError("canonical_train.returned must contain only binary 0/1 values.")

    source_returned = canonical_train[TARGET_COLUMN].astype("int8").reset_index(
        drop=True
    )
    has_pickup = canonical_train["pickup_scheduled_at"].notna().reset_index(drop=True)
    has_reverse_pickup = canonical_train["last_service_event_type"].eq(
        "REVERSE_PICKUP"
    ).reset_index(drop=True)
    workflow_evidence = has_pickup | has_reverse_pickup
    supported_positive = source_returned.eq(1) | workflow_evidence

    labels = pd.Series(pd.NA, index=range(len(canonical_train)), dtype="Int8")
    labels.loc[supported_positive] = 1

    statuses: list[str] = []
    reasons: list[str] = []
    for source_label, pickup, reverse_event in zip(
        source_returned, has_pickup, has_reverse_pickup
    ):
        has_workflow_evidence = bool(pickup or reverse_event)
        if source_label == 1 and has_workflow_evidence:
            statuses.append("positive_source_and_workflow_evidence")
            reasons.append(
                "The supplied returned flag and approved return-workflow evidence "
                "support a customer request; no request timestamp is available."
            )
        elif source_label == 1:
            statuses.append("positive_source_returned_flag")
            reasons.append(
                "The supplied returned flag supports a request under Kestrel's "
                "process definition; no request event or timestamp is available."
            )
        elif has_workflow_evidence:
            statuses.append("positive_approved_return_workflow_evidence")
            reasons.append(
                "A scheduled pickup or reverse-pickup event supports an approved "
                "customer request; a booked pickup may later be cancelled."
            )
        else:
            statuses.append("unknown_no_request_evidence")
            reasons.append(
                "No positive request evidence is present. This is not a confirmed "
                "negative because request history, delivery date, and label cutoff "
                "are unavailable."
            )

    return pd.DataFrame(
        {
            "order_id": canonical_train["order_id"].reset_index(drop=True),
            "source_returned": source_returned,
            "has_pickup_scheduled_at": has_pickup.astype("int8"),
            "has_reverse_pickup_event": has_reverse_pickup.astype("int8"),
            "return_request_label_proxy": labels,
            "label_status": statuses,
            "label_reason": reasons,
            "label_maturity_status": MATURITY_STATUS,
        }
    )


def prepare_return_request_label_audit(data_dir: Path) -> pd.DataFrame:
    """Canonicalize the source training table and audit request-label evidence."""
    train = pd.read_csv(
        _find_one(data_dir, "train.csv"),
        dtype={"order_id": "string"},
    )
    canonical, _, _ = canonicalize_train(train)
    return build_return_request_label_audit(canonical)


def _check_domain(frame: pd.DataFrame, name: str) -> None:
    allowed = {
        "sales_channel": {"app", "web", "marketplace", "partner_outlet"},
        "payment_mode": {"prepaid_upi", "prepaid_card", "cod", "emi"},
        "is_gift": {"Y", "N"},
    }
    for column, values in allowed.items():
        observed = set(frame[column].dropna().unique())
        if not observed.issubset(values) or frame[column].isna().any():
            raise ValueError(
                f"{name}.{column} has missing or unsupported values: "
                f"{sorted(observed.difference(values))}"
            )


def _prepare_features(
    orders: pd.DataFrame,
    products: pd.DataFrame,
    customer_ids: set[str],
    name: str,
) -> pd.DataFrame:
    _require_columns(orders, set(ORDER_COLUMNS), name)
    if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
        raise ValueError(f"{name} order_id values must be present and unique.")
    if not set(orders["customer_id"].dropna()).issubset(customer_ids):
        raise ValueError(f"{name} contains customer IDs missing from customers.csv.")
    if orders["customer_id"].isna().any():
        raise ValueError(f"{name} contains missing customer_id values.")
    _check_domain(orders, name)
    order_times = _parse_order_times(orders, name)

    product_features = products[
        ["sku", "family", "model_name", "launch_date"]
    ].copy()
    if product_features["sku"].isna().any() or product_features["sku"].duplicated().any():
        raise ValueError("products.csv SKU keys must be present and unique.")
    product_features["launch_date"] = pd.to_datetime(
        product_features["launch_date"], errors="raise"
    )
    joined = orders.merge(
        product_features,
        on="sku",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing_skus = joined.loc[joined["_merge"].ne("both"), "sku"].drop_duplicates()
    if not missing_skus.empty:
        raise ValueError(
            f"{name} contains SKU IDs missing from products.csv: "
            f"{missing_skus.head(10).tolist()}"
        )
    joined = joined.drop(columns="_merge")
    if joined["launch_date"].isna().any():
        raise ValueError("products.csv contains missing launch dates for joined SKUs.")

    parsed_order_times = pd.Series(order_times.to_numpy(), index=orders.index)
    product_age = (
        parsed_order_times.reset_index(drop=True)
        - joined["launch_date"].reset_index(drop=True)
    )
    if (product_age.dt.total_seconds() < 0).any():
        raise ValueError(f"{name} contains orders placed before their product launch.")

    output = pd.DataFrame(index=range(len(joined)))
    output["order_id"] = joined["order_id"].reset_index(drop=True)
    output["order_placed_at"] = orders["order_placed_at"].reset_index(drop=True)
    for column in (
        "sales_channel",
        "payment_mode",
        "discount_pct",
        "qty",
        "promised_delivery_days",
        "is_gift",
    ):
        output[column] = joined[column].reset_index(drop=True)

    pincode = joined["delivery_pincode"].reset_index(drop=True)
    if pincode.isna().any():
        raise ValueError(f"{name} contains missing delivery_pincode values.")
    output["delivery_pincode"] = pincode.astype("int64").astype(str).str.zfill(6)
    output["has_default_pincode"] = pincode.eq(0).astype("int8")

    output["order_hour"] = order_times.dt.hour.to_numpy()
    output["order_day_of_week"] = order_times.dt.day_of_week.to_numpy()
    output["order_day"] = order_times.dt.day.to_numpy()
    output["order_month"] = order_times.dt.month.to_numpy()
    output["is_weekend"] = (order_times.dt.day_of_week >= 5).to_numpy(dtype="int8")
    output["sku"] = joined["sku"].reset_index(drop=True)
    output["product_family"] = joined["family"].reset_index(drop=True)
    output["product_model_name"] = joined["model_name"].reset_index(drop=True)
    output["product_age_days"] = product_age.dt.total_seconds().floordiv(86400).astype(
        "int64"
    )

    if output[list(FEATURE_COLUMNS)].isna().any().any():
        missing = output[list(FEATURE_COLUMNS)].isna().sum()
        raise ValueError(
            f"{name} produced missing model features: "
            f"{missing[missing.gt(0)].to_dict()}"
        )
    return output[list(OUTPUT_METADATA_COLUMNS) + list(FEATURE_COLUMNS)]


def prepare_datasets(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load, validate, deduplicate, and prepare model-input tables."""
    paths = {
        "train": _find_one(data_dir, "train.csv"),
        "test": _find_one(data_dir, "test_unlabelled.csv"),
        "customers": _find_one(data_dir, "customers.csv"),
        "products": _find_one(data_dir, "products.csv"),
        "sample": _find_one(data_dir, "sample_submission.csv"),
    }
    train = pd.read_csv(paths["train"], dtype={"order_id": "string"})
    test = pd.read_csv(paths["test"], dtype={"order_id": "string"})
    customers = pd.read_csv(paths["customers"], dtype={"customer_id": "string"})
    products = pd.read_csv(paths["products"], dtype={"sku": "string"})
    sample = pd.read_csv(paths["sample"], dtype={"order_id": "string"})

    _require_columns(customers, REFERENCE_COLUMNS["customers"], "customers.csv")
    _require_columns(products, REFERENCE_COLUMNS["products"], "products.csv")
    if customers["customer_id"].isna().any() or customers["customer_id"].duplicated().any():
        raise ValueError("customers.csv customer_id keys must be present and unique.")
    if not customers["shield_member"].isin({"Y", "N"}).all():
        raise ValueError("customers.csv has missing or unsupported shield_member values.")
    if not set(test.columns).issubset(set(train.columns)):
        raise ValueError("test_unlabelled.csv contains columns not present in train.csv.")
    if TARGET_COLUMN in test.columns:
        raise ValueError("test_unlabelled.csv must not contain the target.")
    if set(train.columns).difference({TARGET_COLUMN}) != set(test.columns):
        raise ValueError(
            "Train and test feature schemas differ beyond the returned target."
        )
    if sample["order_id"].duplicated().any():
        raise ValueError("sample_submission.csv contains duplicate order IDs.")
    if sample["order_id"].tolist() != test["order_id"].tolist():
        raise ValueError(
            "sample_submission order IDs must exactly match test order and row order."
        )

    canonical, duplicate_audit, duplicate_stats = canonicalize_train(train)
    customer_ids = set(customers["customer_id"].dropna().astype(str))
    train_features = _prepare_features(
        canonical, products, customer_ids, "canonical train"
    )
    test_features = _prepare_features(test, products, customer_ids, "test")
    train_features[TARGET_COLUMN] = canonical[TARGET_COLUMN].reset_index(drop=True).astype(
        "int8"
    )
    train_features["label_maturity_status"] = MATURITY_STATUS

    product_index = products.set_index("sku")
    order_dates = _parse_order_times(canonical, "canonical train").dt.normalize()
    signup_dates = pd.to_datetime(
        canonical["customer_id"].map(
            customers.set_index("customer_id")["signup_date"]
        ),
        errors="raise",
    )
    signup_after_order_count = int((signup_dates.to_numpy() > order_dates.to_numpy()).sum())
    catalog_expected = (
        canonical["sku"].map(product_index["list_price_inr"])
        * canonical["qty"]
        * (1 - canonical["discount_pct"] / 100)
    )
    october = _parse_order_times(canonical, "canonical train").dt.to_period("M").eq(
        "2025-10"
    )
    amount_ratio = canonical.loc[october, "order_value_inr"] / catalog_expected.loc[
        october
    ]
    stats = {
        **duplicate_stats,
        "train_label_counts": {
            str(key): int(value)
            for key, value in canonical[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "customer_reference_rows": len(customers),
        "product_reference_rows": len(products),
        "train_customer_signup_after_order": signup_after_order_count,
        "train_joined_customer_ids": int(canonical["customer_id"].nunique()),
        "test_joined_customer_ids": int(test["customer_id"].nunique()),
        "train_october_2025_rows": int(october.sum()),
        "train_october_ratio_values": sorted(
            float(value) for value in amount_ratio.round(8).unique()
        ),
        "train_latest_order_date": str(
            _parse_order_times(canonical, "canonical train").max()
        ),
        "test_latest_order_date": str(_parse_order_times(test, "test").max()),
        "product_features_used": [
            "family",
            "model_name",
            "launch_date (derived product_age_days)",
        ],
        "customer_reference_features_used": [],
    }
    return train_features, test_features, duplicate_audit, stats


def _write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output {path}; pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _ensure_outside_data_dir(path: Path, data_dir: Path) -> None:
    try:
        path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        return
    raise ValueError(
        f"Refusing to write generated output inside the supplied datasets directory: {path}"
    )


def _prepare_product_reference(data_dir: Path) -> pd.DataFrame:
    """Keep only product fields required to build safe model features at serving time."""
    products = pd.read_csv(
        _find_one(data_dir, "products.csv"),
        dtype={"sku": "string"},
    )
    required = {"sku", "family", "model_name", "launch_date"}
    _require_columns(products, required, "products.csv")
    reference = products[["sku", "family", "model_name", "launch_date"]].copy()
    if reference.isna().any().any() or reference["sku"].duplicated().any():
        raise ValueError(
            "Product serving reference requires complete rows and unique SKU keys."
        )
    reference["launch_date"] = pd.to_datetime(
        reference["launch_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    return reference.sort_values("sku", kind="stable").reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Prepare point-in-time-safe returns modelling CSVs."
    )
    parser.add_argument("--data-dir", type=Path, default=root / "datasets")
    parser.add_argument("--output-dir", type=Path, default=root / "derived_data")
    parser.add_argument("--reports-dir", type=Path, default=root / "reports")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the generated CSVs if they already exist.",
    )
    parser.add_argument(
        "--return-request-label-audit-only",
        action="store_true",
        help=(
            "Write only the derived one-sided return-request label audit; "
            "unknown outcomes remain unlabelled."
        ),
    )
    args = parser.parse_args()

    if args.return_request_label_audit_only:
        audit_path = args.output_dir / "return_request_label_audit.csv"
        _ensure_outside_data_dir(audit_path, args.data_dir)
        audit = prepare_return_request_label_audit(args.data_dir)
        _write_csv(audit, audit_path, args.overwrite)
        positives = int(audit["return_request_label_proxy"].eq(1).sum())
        unknown = int(audit["return_request_label_proxy"].isna().sum())
        print(
            f"Wrote {len(audit)} request-label audit rows: {positives} "
            f"positive-evidence labels and {unknown} unknown; no negative "
            "labels were inferred."
        )
        return

    outputs = [
        args.output_dir / "train_modeling.csv",
        args.output_dir / "test_modeling.csv",
        args.output_dir / "product_reference.csv",
        args.reports_dir / "duplicate_audit.csv",
    ]
    for output_path in outputs:
        _ensure_outside_data_dir(output_path, args.data_dir)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing outputs: {existing}. "
            "Pass --overwrite to replace them."
        )

    train, test, duplicate_audit, stats = prepare_datasets(args.data_dir)
    product_reference = _prepare_product_reference(args.data_dir)
    _write_csv(train, outputs[0], args.overwrite)
    _write_csv(test, outputs[1], args.overwrite)
    _write_csv(product_reference, outputs[2], args.overwrite)
    _write_csv(duplicate_audit, outputs[3], args.overwrite)
    print(f"Prepared {len(train)} canonical train rows and {len(test)} test rows.")
    print(
        f"Audited {stats['discarded_rows']} discarded exact partner-feed duplicates; "
        "label maturity remains unverified."
    )
    print(
        "October 2025 canonical rows excluded from amount features: "
        f"{stats['train_october_2025_rows']}; observed amount/catalog factors: "
        f"{stats['train_october_ratio_values']}."
    )


if __name__ == "__main__":
    main()
