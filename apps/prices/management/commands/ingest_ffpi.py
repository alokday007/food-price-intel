"""Ingest the FAO Food Price Index workbook into PriceIndexMonthly (SPEC §5).

Reads ONLY the two monthly sheets and idempotently upserts on
(commodity_group, period) so FAO's monthly revisions UPDATE in place instead of
duplicating rows. Annual sheets are deliberately ignored — they are yearly and
must never enter a monthly table.

A note on the workbook's real shape (confirmed programmatically, not assumed):
the nominal sheet ``Indices_Monthly`` and the real sheet ``Indices_Monthly_Real``
do NOT share a layout. The nominal sheet's date column is ``Date`` with bare
sub-index names (``Meat``, ``Dairy``, …); the real sheet's date column is
``Month`` with a `` Price Index`` suffix on every column (``Meat Price Index``).
So each sheet gets its own explicit column map below, and a missing expected
column fails loudly rather than being skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from apps.catalog.models import CommodityGroup, DataSource
from apps.prices.models import PriceIndexMonthly

SOURCE_NAME = "FAO FFPI"

# Header (row 3 in Excel) lives at pandas header index 2 on both monthly sheets.
HEADER_ROW = 2

# Soft sanity band for an index value (SPEC §5). Outside this we warn but still
# store; only clearly-corrupt values (<=0 or absurdly large) abort the run.
SANE_MIN, SANE_MAX = Decimal("20"), Decimal("300")
HARD_MIN, HARD_MAX = Decimal("0"), Decimal("1000")


@dataclass
class SheetSpec:
    """How to read one monthly sheet into (group_code -> column) records."""

    sheet_name: str
    date_col: str
    # Excel column header -> CommodityGroup.code
    value_cols: dict[str, str]


# The nominal sheet matches the documented structure exactly.
NOMINAL = SheetSpec(
    sheet_name="Indices_Monthly",
    date_col="Date",
    value_cols={
        "Food Price Index": "OVERALL",
        "Meat": "MEAT",
        "Dairy": "DAIRY",
        "Cereals": "CEREALS",
        "Oils": "OILS",
        "Sugar": "SUGAR",
    },
)

# The real sheet uses a different date column and `` Price Index`` suffixes.
REAL = SheetSpec(
    sheet_name="Indices_Monthly_Real",
    date_col="Month",
    value_cols={
        "Food Price Index": "OVERALL",
        "Meat Price Index": "MEAT",
        "Dairy Price Index": "DAIRY",
        "Cereals Price Index": "CEREALS",
        "Oils Price Index": "OILS",
        "Sugar Price Index": "SUGAR",
    },
)


@dataclass
class Record:
    """A single (group, period) observation accumulated across both sheets."""

    value_nominal: Decimal
    value_real: Decimal | None = None


def _to_period(value) -> date | None:
    """Parse a cell to the first day of its month, or None if it isn't a date."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            ts = pd.to_datetime(text)  # handles "YYYY-MM" and full dates
        except (ValueError, TypeError):
            return None
        return date(ts.year, ts.month, 1)
    # pandas Timestamp / datetime / date
    try:
        return date(value.year, value.month, 1)
    except AttributeError:
        return None


def _to_decimal(value) -> Decimal | None:
    """Coerce a numeric cell to Decimal; None for blank/footnote/non-numeric."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return Decimal(text)
        except InvalidOperation:
            return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


@dataclass
class SheetReport:
    """STEP 1 confirmation payload for one sheet."""

    found: bool
    header_row: int
    columns: list[str] = field(default_factory=list)
    date_dtype: str = ""
    value_dtype: str = ""
    head: list[tuple] = field(default_factory=list)
    tail: list[tuple] = field(default_factory=list)
    rows_parsed: int = 0


class Command(BaseCommand):
    help = "Ingest the FAO FFPI workbook (nominal + real) into PriceIndexMonthly."

    def add_arguments(self, parser):
        parser.add_argument(
            "--file", required=True, help="Path to the FAO FFPI .xlsx workbook."
        )

    def handle(self, *args, **options):
        path = options["file"]

        # Resolve dimension rows up front; never recreate them here (SPEC §5).
        try:
            source = DataSource.objects.get(name=SOURCE_NAME)
        except DataSource.DoesNotExist:
            raise CommandError(
                f"DataSource '{SOURCE_NAME}' not found — run seed_catalog first."
            )
        groups = {g.code: g for g in CommodityGroup.objects.all()}
        missing_groups = set(NOMINAL.value_cols.values()) - groups.keys()
        if missing_groups:
            raise CommandError(
                f"Missing CommodityGroup rows {sorted(missing_groups)} — run seed_catalog."
            )

        try:
            sheet_names = pd.ExcelFile(path).sheet_names
        except FileNotFoundError:
            raise CommandError(f"Workbook not found: {path}")

        # --- STEP 1: confirm both monthly sheets, fail loudly on problems ---
        records: dict[tuple[str, date], Record] = {}
        nominal_report = self._read_sheet(path, sheet_names, NOMINAL, records, is_real=False)
        real_report = self._read_sheet(path, sheet_names, REAL, records, is_real=True)
        self._print_confirmation(NOMINAL, nominal_report)
        self._print_confirmation(REAL, real_report)

        # --- STEP 3: idempotent upsert ---
        created, updated, out_of_band = self._upsert(records, groups, source)

        source.last_ingested_at = timezone.now()
        source.save(update_fields=["last_ingested_at"])

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("FFPI ingest complete:"))
        self.stdout.write(f"  records parsed : {len(records)}")
        self.stdout.write(f"  created        : {created}")
        self.stdout.write(f"  updated        : {updated}")
        if out_of_band:
            self.stdout.write(
                self.style.WARNING(
                    f"  out-of-band    : {out_of_band} value(s) outside "
                    f"{SANE_MIN}..{SANE_MAX} (stored anyway)"
                )
            )
        self.stdout.write(
            f"  source '{SOURCE_NAME}'.last_ingested_at = {source.last_ingested_at:%Y-%m-%d %H:%M:%S}"
        )

    def _read_sheet(self, path, sheet_names, spec: SheetSpec, records, *, is_real: bool) -> SheetReport:
        """Parse one sheet into the shared records dict; build its confirm report."""
        if spec.sheet_name not in sheet_names:
            raise CommandError(f"Required sheet '{spec.sheet_name}' not found in workbook.")

        df = pd.read_excel(path, sheet_name=spec.sheet_name, header=HEADER_ROW)

        # Fail loudly on any missing expected column (date or a value column).
        expected = [spec.date_col, *spec.value_cols.keys()]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise CommandError(
                f"Sheet '{spec.sheet_name}' is missing expected column(s): {missing}. "
                f"Found: {list(df.columns)[:10]}"
            )

        report = SheetReport(
            found=True,
            header_row=HEADER_ROW,
            columns=expected,
            date_dtype=str(df[spec.date_col].dtype),
            value_dtype=str(df["Food Price Index"].dtype),
        )

        parsed = 0
        for _, row in df.iterrows():
            period = _to_period(row[spec.date_col])
            if period is None:
                continue  # blank / footnote / non-date row
            row_had_value = False
            for col, code in spec.value_cols.items():
                val = _to_decimal(row[col])
                if val is None:
                    continue  # blank/footnote cell for this group
                row_had_value = True
                key = (code, period)
                if is_real:
                    rec = records.get(key)
                    if rec is not None:  # real attaches to an existing nominal row
                        rec.value_real = val
                else:
                    records[key] = Record(value_nominal=val)
            if row_had_value:
                parsed += 1

        report.rows_parsed = parsed

        # Sample the OVERALL series (sorted) for the confirm head/tail.
        overall = sorted(
            (
                (p, r.value_nominal if not is_real else r.value_real)
                for (c, p), r in records.items()
                if c == "OVERALL"
            )
        )
        report.head = overall[:3]
        report.tail = overall[-3:]
        return report

    def _print_confirmation(self, spec: SheetSpec, rep: SheetReport):
        self.stdout.write(self.style.MIGRATE_HEADING(f"[STEP 1] Sheet '{spec.sheet_name}'"))
        self.stdout.write(f"  found            : {'yes' if rep.found else 'NO'}")
        self.stdout.write(f"  header row       : pandas header={rep.header_row} (Excel row {rep.header_row + 1})")
        self.stdout.write(f"  date column      : {spec.date_col} (dtype {rep.date_dtype})")
        self.stdout.write(f"  value columns    : {list(spec.value_cols.keys())}")
        self.stdout.write(f"  'Food Price Index' dtype: {rep.value_dtype}")
        self.stdout.write(f"  rows parsed      : {rep.rows_parsed}")
        self.stdout.write("  first 3 OVERALL  : " + ", ".join(f"{p:%Y-%m}={v}" for p, v in rep.head))
        self.stdout.write("  last 3 OVERALL   : " + ", ".join(f"{p:%Y-%m}={v}" for p, v in rep.tail))

    @transaction.atomic
    def _upsert(self, records, groups, source) -> tuple[int, int, int]:
        created = updated = out_of_band = 0
        for (code, period), rec in records.items():
            nominal = rec.value_nominal
            if not (HARD_MIN < nominal < HARD_MAX):
                raise CommandError(
                    f"Corrupt value for {code} {period}: {nominal} outside "
                    f"({HARD_MIN}, {HARD_MAX})."
                )
            if not (SANE_MIN <= nominal <= SANE_MAX):
                out_of_band += 1
            _, was_created = PriceIndexMonthly.objects.update_or_create(
                commodity_group=groups[code],
                period=period,
                defaults={
                    "value_nominal": nominal,
                    "value_real": rec.value_real,
                    "source": source,
                },
            )
            created += was_created
            updated += not was_created
        return created, updated, out_of_band
