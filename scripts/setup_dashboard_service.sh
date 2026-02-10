#!/bin/bash
# Setup DeFi Dashboard as a systemd service
# Run with: sudo bash scripts/setup_dashboard_service.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
USER=$(logname)

echo "=========================================="
echo "DeFi Dashboard Service Setup"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo "User: $USER"
echo ""

# Create service file
echo "Creating systemd service file..."
cat > /etc/systemd/system/defi-dashboard.service << EOF
[Unit]
Description=DeFi Yield Rebalancing Dashboard
After=network.target postgresql.service
Documentation=https://github.com/your-repo/Defi-Yield-R&D

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin:/usr/local/bin:/usr/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$PROJECT_DIR/venv/bin/python -m streamlit run dashboard/app.py --server.headless=true --server.port=8501 --browser.gatherUsageStats=false --server.address=0.0.0.0
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Service file created"

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable service
echo "Enabling service..."
systemctl enable defi-dashboard

# Start service
echo "Starting service..."
systemctl start defi-dashboard

# Wait for service to start
sleep 3

# Check status
echo ""
echo "=========================================="
echo "Service Status:"
echo "=========================================="
systemctl status defi-dashboard --no-pager || true

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Dashboard URL: http://localhost:8501"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status defi-dashboard    # Check status"
echo "  sudo systemctl restart defi-dashboard   # Restart"
echo "  sudo systemctl stop defi-dashboard      # Stop"
echo "  sudo journalctl -u defi-dashboard -f    # View logs"
echo ""
echo "Dashboard will auto-start on system boot!"
echo "=========================================="
