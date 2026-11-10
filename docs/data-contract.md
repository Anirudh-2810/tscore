# tscore data contract — v1

**Status:** active (Week 1). **Owner:** tscore. **Consumers:** StockOffline
(`inventory-system`), tscore baselines/backtest (Week 2+).

Rules for this file:

- A breaking change (column rename/remove, type change, semantic change) bumps
  the version and updates producer and consumer **in the same change**.
- Anything not written here is not part of the contract.
- Confidence and money claims are never inferred — unknown stays unknown.

## 1. Sales CSV (input)

The single sales format. Shops export it; `scripts/generate_fake_sales.py`
fakes it; StockOffline's `import-sales` consumes it.

| Column        | Type              | Rules                                                        |
|---------------|-------------------|--------------------------------------------------------------|
| `item`        | string, non-empty | SKU preferred; barcode or exact product name also resolves   |
| `date`        | date, `YYYY-MM-DD`| ISO 8601, no time component, not in the future               |
| `quantity`    | integer           | >= 1 (a sale of 0 or negative is invalid)                    |
| `price`       | number            | unit price at time of sale, >= 0, decimal point (`.`)        |

```csv
item,date,quantity,price
PARACETAMOL-500,2026-10-01,3,30.00
CETIRIZINE-10,2026-10-01,1,45.50
PARACETAMOL-500,2026-10-02,5,30.00
```

**File rules**

- Encoding: UTF-8 (UTF-8 with BOM accepted — Excel exports carry a BOM).
- Header row required. The four columns must be present; column order may vary,
  extra columns are ignored (they are not part of the contract).
- One row = one sale line. Rows are not required to be sorted.
- Line endings: LF or CRLF both accepted.

**Import semantics (StockOffline `import-sales`)**

- Row resolution order: `sku` exact → `barcode` exact → `name` case-insensitive
  (trimmed). First hit wins.
- Unknown `item` → row skipped and reported (never silently dropped, never
  auto-created unless `--create-missing` is passed).
- Import records **historical** sale transactions dated by `date` and
  **does not mutate current stock** — current stock is what the shop counted,
  history is what the shop sold. Double-counting past sales would corrupt both.
- Invalid rows (bad date, qty < 1, negative price) are skipped with a row
  number in the report. A structurally broken file (missing header columns)
  is a hard error: nothing is imported.

## 2. Forecast output (tscore → StockOffline report)

One row per item that needs attention. Written by the baseline/model, read by
the weekly report and the Forecast screen.

| Column             | Type   | Rules                                                        |
|--------------------|--------|--------------------------------------------------------------|
| `item`             | string | must resolve against the shop's catalog (same rules as sales)|
| `predicted_demand` | number | units expected to sell over the forecast horizon, >= 0       |
| `suggested_reorder`| number | units to order now, integer >= 0, **0 means "don't order"**  |
| `confidence`       | number | 0.0 – 1.0, 2+ decimal places                                 |

```csv
item,predicted_demand,suggested_reorder,confidence
PARACETAMOL-500,42,50,0.73
CETIRIZINE-10,18,0,0.61
```

**Invariants (Week 3 guards will enforce, tests prove)**

- `suggested_reorder` never exceeds the max-order, budget-cap, or
  last-order-multiple guards — a row violating a guard is a bug, not a suggestion.
- `confidence` is the producer's stated calibration, not a claim of accuracy;
  backtest error is measured separately (Week 2) and reported beside it.
- Missing forecast for a catalog item is allowed (it means "no opinion"), but a
  forecast for a non-existent item is invalid.

## 3. Out of scope (v1)

- Batch/expiry fields (pharmacy expiry tracker — later feature)
- Supplier lead times as structured columns (they live in StockOffline's
  product record: `supplier`, and later a lead-time field)
- Multi-currency: prices are the shop's single local currency, plain decimal
