# Data Model: CPU Model Compatibility

All fields are additive, response-derived data; no new persisted schema exists.

## Machine CPU Snapshot

| Field | Type | Rule |
|---|---|---|
| `architecture` | string/null | Service-host CPU architecture. |
| `logical_cores` | integer/null | Positive logical CPU count. |
| `available_ram_gb` | number/null | Current available RAM. |
| `capacity_margin_gb` | number | Non-negative configured margin. |
| `usable_ram_gb` | number/null | `max(0, available - margin)`. |
| `observed_at` | ISO-8601 | Sampling time. |

## Model Resource Profile

Raw llmfit data is retained. Only verified explicit aliases populate optional `cpu_architecture_requirement`, `min_cpu_cores`, and runtime placement/offload evidence. A real llmfit/raw field alias must be verified against captured data before it is named in implementation work; unverified fields do not produce an offload band. `memory_required_gb` and `runtime` retain their current normalized meanings.

## Runtime Offload Evidence Contract

The selected serving runtime is the normalized existing `model.runtime` associated with the catalogue or recommendation record. This feature does not add a runtime selector or alter external runtime placement.

An offload band is available only when verified runtime- or catalogue-provided evidence normalizes to this object:

```json
{
  "runtime": "<normalized model.runtime>",
  "band": "low | medium | high",
  "observed_at": "<ISO-8601 timestamp>",
  "source": "runtime_reported | catalogue_metadata"
}
```

`band` is accepted only as one of the three literal values above. Evidence is stale when `observed_at` is more than 300 seconds old at response enrichment time. Missing, malformed, mismatched-runtime, stale, or unverified evidence returns `status: "unavailable"` and `band: null`. No model name, parameter count, VRAM ratio, or runtime default may infer a band. Implementation tasks must verify a real llmfit/raw alias before naming or accepting it; until then, the result is unavailable.

## `cpu_compatibility`

| Field | Type | Rule |
|---|---|---|
| `status` | compatible/incompatible/unknown | Compatible only with complete known requirements and passing capacity; incompatible on a known failure; unknown otherwise. |
| `reason` | string | First limiting or missing-input explanation. |
| `capacity_basis` | object | Machine snapshot and model requirements used. |
| `calculated_at` | ISO-8601 | Response-cycle time. |
| `input_state` | complete/missing_model_data/missing_machine_data/stale | Input availability. |

Decision order: missing RAM/architecture/core facts or required model inputs -> unknown; memory above usable RAM, architecture mismatch, or inadequate cores -> incompatible; otherwise compatible.

## `cpu_offload_likelihood`

| Field | Type | Rule |
|---|---|---|
| `status` | available/unavailable | Available only with verified runtime-specific evidence. |
| `band` | low/medium/high/null | Non-null only for available; never a percentage. |
| `reason` | string | Evidence or unavailability explanation. |
| `runtime` | string/null | Runtime the result applies to. |
| `source` | runtime_reported/catalogue_metadata/unavailable | Provenance. |
| `calculated_at` | ISO-8601 | Response-cycle time. |

Backend-managed, unreported, malformed, mismatched-runtime, stale, and unverified evidence returns unavailable/null. The UI confirms only `high`.

## Lifecycle

Model resource profile + machine snapshot produces one compatibility assessment. Model runtime + runtime-specific evidence produces one likelihood. Both travel with responses and remain within the existing assigned `catalog_model` snapshot; they refresh next response and need no migration.
