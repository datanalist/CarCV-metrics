#!/bin/bash
# Deploy project to both servers in parallel, then run setup + download

set -e
DEPLOY_DIR="$(dirname "$0")/.."

echo "=== Deploying CARS evaluation project to servers ==="

deploy_server() {
    local server="$1"
    local claude_md="$2"
    echo "[$server] Starting deployment..."

    # Sync project files
    rsync -avz --progress \
        --exclude=".git" --exclude="venv" --exclude="__pycache__" \
        --exclude="*.pyc" --exclude="data" --exclude="models" \
        --exclude="results" --exclude="plots" \
        "$DEPLOY_DIR/" "$server:~/cars-eval/"

    # Copy server-specific CLAUDE.md
    scp "$DEPLOY_DIR/$claude_md" "$server:~/cars-eval/CLAUDE.md"

    # Run setup
    echo "[$server] Running setup..."
    ssh "$server" "cd ~/cars-eval && bash scripts/setup_server.sh"

    echo "[$server] Deployment complete"
}

# Deploy in parallel
deploy_server "qudata2" "CLAUDE_qudata2.md" &
PID2=$!
deploy_server "qudata5" "CLAUDE_qudata5.md" &
PID5=$!

wait $PID2
echo "[qudata2] Setup finished"
wait $PID5
echo "[qudata5] Setup finished"

echo ""
echo "=== Both servers ready. Next steps: ==="
echo "  qudata2: ssh qudata2 'cd ~/cars-eval && bash scripts/download_models.sh && bash scripts/download_datasets_qudata2.sh'"
echo "  qudata5: ssh qudata5 'cd ~/cars-eval && bash scripts/download_models.sh && bash scripts/download_datasets_qudata5.sh'"
echo ""
echo "  Or run evaluation directly:"
echo "  qudata2: ssh qudata2 'cd ~/cars-eval && source venv/bin/activate && python evaluation/evaluate.py --models trafficcamnet vehiclemakenet'"
echo "  qudata5: ssh qudata5 'cd ~/cars-eval && source venv/bin/activate && python evaluation/evaluate.py --models vehicletypenet lpdnet lprnet'"
