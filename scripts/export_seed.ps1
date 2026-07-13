# Exports a snapshot of the analytics schema from the running local
# Postgres into seed/03_analytics_seed.sql, which docker-compose uses
# to seed the demo stack. Re-run after any pipeline changes you want
# reflected in the demo.
#
# Usage (from repo root, with civiclens_pg running):
#   .\scripts\export_seed.ps1

New-Item -ItemType Directory -Force -Path seed | Out-Null

docker exec civiclens_pg pg_dump -U civiclens -d civiclens `
    --schema=analytics `
    --no-owner --no-privileges `
    | Out-File -Encoding utf8 seed/03_analytics_seed.sql

Write-Host "Wrote seed/03_analytics_seed.sql"
Write-Host ("Size: {0:N0} KB" -f ((Get-Item seed/03_analytics_seed.sql).Length / 1KB))