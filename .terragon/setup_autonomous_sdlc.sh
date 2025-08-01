#!/bin/bash
# Terragon Autonomous SDLC Setup Script
# Sets up perpetual value discovery and execution system

set -e

echo "🚀 TERRAGON AUTONOMOUS SDLC SETUP"
echo "=================================="

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
TERRAGON_DIR="$REPO_ROOT/.terragon"

echo "📁 Repository: $REPO_ROOT"
echo "📁 Terragon Directory: $TERRAGON_DIR"

# Ensure .terragon directory exists
mkdir -p "$TERRAGON_DIR"

# Make scripts executable
chmod +x "$TERRAGON_DIR/value_discovery_simple.py"
chmod +x "$TERRAGON_DIR/perpetual_executor.py"

echo "✅ Scripts made executable"

# Run initial value discovery
echo "🔍 Running initial value discovery..."
python3 "$TERRAGON_DIR/value_discovery_simple.py"

echo "📊 Checking executor status..."
python3 "$TERRAGON_DIR/perpetual_executor.py" --status

echo ""
echo "🎯 AUTONOMOUS SDLC SETUP COMPLETE"
echo "=================================="
echo ""
echo "📋 Available Commands:"
echo "  • Run single cycle:     python3 .terragon/perpetual_executor.py --single"
echo "  • Force execution:      python3 .terragon/perpetual_executor.py --single --force"
echo "  • Continuous mode:      python3 .terragon/perpetual_executor.py --continuous 24"
echo "  • Check status:         python3 .terragon/perpetual_executor.py --status"
echo "  • Setup cron:           python3 .terragon/perpetual_executor.py --setup-cron"
echo ""
echo "📅 Scheduling Options:"
echo "  • Add to crontab for automatic execution"
echo "  • Use continuous mode for long-running operations"
echo "  • Integrate with CI/CD for post-merge execution"
echo ""
echo "📈 Value Discovery Active:"
echo "  • BACKLOG.md contains discovered opportunities"
echo "  • Scoring based on WSJF + ICE + Technical Debt"
echo "  • Adaptive weights for advanced repository maturity"
echo ""
echo "🔄 Next Steps:"
echo "  1. Review BACKLOG.md for discovered value items"
echo "  2. Execute highest value item: $(python3 .terragon/value_discovery_simple.py 2>/dev/null | grep "Next Best Value" | head -1 || echo "Check BACKLOG.md")"
echo "  3. Set up scheduling for continuous value discovery"
echo "  4. Monitor execution outcomes and refine scoring model"
echo ""
echo "✨ Autonomous SDLC enhancement is now operational!"