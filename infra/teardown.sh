#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# teardown.sh – Delete SLD BOM Extraction Azure infrastructure
# Usage:
#   ./infra/teardown.sh          # dev (default)
#   ./infra/teardown.sh prod     # production
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

ENV="${1:-dev}"
RESOURCE_GROUP="rg-sldbom-${ENV}"

echo "── WARNING: This will permanently delete all resources in:"
echo "   Resource Group: ${RESOURCE_GROUP}"
echo ""

read -rp "Are you sure? Type the resource group name to confirm: " confirm
if [[ "$confirm" != "$RESOURCE_GROUP" ]]; then
  echo "Aborted."
  exit 0
fi

echo ""
echo "── Deleting resource group: ${RESOURCE_GROUP}..."
az group delete --name "$RESOURCE_GROUP" --yes --no-wait

echo "── Deletion initiated (running in background)."
echo "   Monitor: az group show --name ${RESOURCE_GROUP} --query properties.provisioningState"
