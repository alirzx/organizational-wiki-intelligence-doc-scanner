#!/usr/bin/env bash
set -euo pipefail

EXAMPLE="${1:-.env.example}"
ENV_FILE="${2:-.env}"

if [[ ! -f "$EXAMPLE" ]]; then
    echo "ERROR: missing $EXAMPLE"
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: missing $ENV_FILE"
    exit 1
fi

extract_keys() {
    sed \
        -e 's/\r$//' \
        -e 's/^[[:space:]]*//' \
        -e '/^#/d' \
        -e '/^[[:space:]]*$/d' \
        "$1" |
    grep -E '^(export[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=' |
    sed -E 's/^export[[:space:]]+//' |
    sed -E 's/[[:space:]]*=.*$//' |
    sort -u
}

example_keys="$(mktemp)"
env_keys="$(mktemp)"

trap 'rm -f "$example_keys" "$env_keys"' EXIT

extract_keys "$EXAMPLE" > "$example_keys"
extract_keys "$ENV_FILE" > "$env_keys"

echo
echo "============================================================"
echo "Wiki Hami environment comparison"
echo "============================================================"
echo "Example : $EXAMPLE"
echo "Actual  : $ENV_FILE"
echo

echo ">>> Keys in .env.example but MISSING from .env"
missing="$(comm -23 "$example_keys" "$env_keys" || true)"

if [[ -n "$missing" ]]; then
    echo "$missing"
else
    echo "None"
fi

echo
echo ">>> Keys in .env but NOT documented in .env.example"
extra="$(comm -13 "$example_keys" "$env_keys" || true)"

if [[ -n "$extra" ]]; then
    echo "$extra"
else
    echo "None"
fi

echo
echo ">>> Summary"
echo "Example keys : $(wc -l < "$example_keys")"
echo "Actual keys  : $(wc -l < "$env_keys")"
echo "Missing      : $(comm -23 "$example_keys" "$env_keys" | wc -l)"
echo "Undocumented : $(comm -13 "$example_keys" "$env_keys" | wc -l)"
echo
