set -euo pipefail
cd "$(dirname "$0")/.."  # về root project

PY=/home/antn/miniconda3/envs/myrag/bin/python
WEB_SCRIPT=scrape/scrape_web.py
PDF_SCRIPT=scrape/scrape_pdf.py
SEEDS=scrape/seeds.yaml
OUT_DIR=data/raw

mkdir -p "$OUT_DIR"

ALL_TASKS=(
    "web  vnu_main       high        1.0"
    "web  vnu_main       med         1.0"
    "web  uet            high        1.5"
    "web  uet            med         1.5"
    "web  uet            subdomains  1.5"
    "web  uet_admissions high        1.5"
    "web  uet_qac        high        1.5"
    "web  uet_ai_institute high      1.5"
    "web  uet_curriculum  high      1.5"
    "web  uet_scholarship high       1.5"
    "web  uet_tuition     high       1.5"
    "web  wikipedia      high        0.5"
    "pdf  pdf_documents  high        2.0"
)

if [[ $# -eq 2 ]]; then
    SECTION="$1"
    TIER="$2"
    TASKS=()
    for t in "${ALL_TASKS[@]}"; do
        read -r typ s tr rate <<< "$t"
        if [[ "$s" == "$SECTION" && "$tr" == "$TIER" ]]; then
            TASKS+=("$t")
        fi
    done
    if [[ ${#TASKS[@]} -eq 0 ]]; then
        echo "ERROR: section=$SECTION tier=$TIER không có trong task list"
        exit 1
    fi
else
    TASKS=("${ALL_TASKS[@]}")
fi

echo "═══════════════════════════════════════════════"
echo "  FULL CRAWL — VNU/UET Knowledge Resource"
echo "═══════════════════════════════════════════════"
echo "  Python:   $PY"
echo "  Seeds:    $SEEDS"
echo "  Output:   $OUT_DIR/"
echo "  Tasks:    ${#TASKS[@]} (section/tier)"
echo "═══════════════════════════════════════════════"
echo ""

TOTAL_START=$(date +%s)
ALL_OK=true

for task in "${TASKS[@]}"; do
    read -r typ section tier rate <<< "$task"
    out_file="${OUT_DIR}/${section}_${tier}.jsonl"

    if [[ "$typ" == "pdf" ]]; then
        SCRIPT="$PDF_SCRIPT"
    else
        SCRIPT="$WEB_SCRIPT"
    fi

    echo "┌─── [${typ}] ${section}/${tier} (rate=${rate}s) ───"
    if $PY "$SCRIPT" \
        --seeds "$SEEDS" \
        --section "$section" \
        --tier "$tier" \
        --rate "$rate" \
        --out "$out_file"; then
        n_docs=$(wc -l < "$out_file" 2>/dev/null || echo 0)
        size=$(du -h "$out_file" 2>/dev/null | cut -f1)
        echo "└── ✓ ${n_docs} docs | ${size}"
    else
        echo "└── ✗ FAILED"
        ALL_OK=false
    fi
    echo ""
done

TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))

echo "═══════════════════════════════════════════════"
echo "  SUMMARY (elapsed: ${ELAPSED}s)"
echo "═══════════════════════════════════════════════"
ls -lh "$OUT_DIR"/*.jsonl 2>/dev/null
echo ""
echo "Total docs across all files:"
wc -l "$OUT_DIR"/*.jsonl 2>/dev/null | tail -1
echo ""
echo "Total content (chars):"
$PY -c "
import json, glob
total=0; ndocs=0
for f in sorted(glob.glob('$OUT_DIR/*.jsonl')):
    for line in open(f):
        d=json.loads(line); total+=d['content_length']; ndocs+=1
print(f'  {total:,} chars from {ndocs} docs (avg {total//max(ndocs,1):,}/doc)')
"

if $ALL_OK; then
    echo ""
    echo "═══ ALL DONE ✓ ═══"
    exit 0
else
    echo ""
    echo "═══ DONE WITH ERRORS ✗ — check log above ═══"
    exit 1
fi
