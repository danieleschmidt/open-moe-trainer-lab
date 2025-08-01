#!/usr/bin/env python3
"""
Terragon Perpetual SDLC Executor
Continuous autonomous enhancement with scheduled execution
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class PerpetualExecutor:
    """Perpetual execution engine for autonomous SDLC enhancement."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.terragon_dir = self.repo_path / ".terragon"
        self.state_file = self.terragon_dir / "executor_state.json"
        self.discovery_script = self.terragon_dir / "value_discovery_simple.py"
        
        self.state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load executor state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "last_discovery": None,
            "last_execution": None,
            "execution_count": 0,
            "continuous_mode": False,
            "schedule": {
                "immediate_after_merge": True,
                "hourly_security_scan": True,
                "daily_comprehensive": True,
                "weekly_deep_analysis": True,
                "monthly_recalibration": True
            }
        }
    
    def _save_state(self):
        """Save current executor state."""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def run_value_discovery(self) -> Dict:
        """Execute value discovery cycle and return results."""
        print(f"🔍 Running value discovery at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            result = subprocess.run(
                [sys.executable, str(self.discovery_script)],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                self.state["last_discovery"] = datetime.now().isoformat()
                self.state["execution_count"] += 1
                self._save_state()
                
                return {
                    "success": True,
                    "output": result.stdout,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def check_git_changes(self) -> bool:
        """Check if there are new commits since last execution."""
        try:
            # Get latest commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                current_commit = result.stdout.strip()
                last_commit = self.state.get("last_commit_processed")
                
                if last_commit != current_commit:
                    self.state["last_commit_processed"] = current_commit
                    return True
                    
        except Exception:
            pass
        
        return False
    
    def should_run_discovery(self, force: bool = False) -> str:
        """Determine if discovery should run based on schedule."""
        if force:
            return "forced_execution"
        
        now = datetime.now()
        last_discovery = self.state.get("last_discovery")
        
        if not last_discovery:
            return "initial_run"
        
        last_run = datetime.fromisoformat(last_discovery)
        time_since_last = now - last_run
        
        # Check for git changes (immediate execution)
        if self.state["schedule"]["immediate_after_merge"] and self.check_git_changes():
            return "git_changes_detected"
        
        # Hourly security scan
        if self.state["schedule"]["hourly_security_scan"] and time_since_last > timedelta(hours=1):
            return "hourly_security_scan"
        
        # Daily comprehensive analysis
        if self.state["schedule"]["daily_comprehensive"] and time_since_last > timedelta(days=1):
            return "daily_comprehensive"
        
        # Weekly deep analysis
        if self.state["schedule"]["weekly_deep_analysis"] and time_since_last > timedelta(days=7):
            return "weekly_analysis"
        
        # Monthly recalibration
        if self.state["schedule"]["monthly_recalibration"] and time_since_last > timedelta(days=30):
            return "monthly_recalibration"
        
        return ""
    
    def execute_highest_value_item(self) -> Dict:
        """Execute the highest value item from the backlog."""
        backlog_file = self.repo_path / "BACKLOG.md"
        
        if not backlog_file.exists():
            return {"success": False, "error": "No backlog file found"}
        
        try:
            with open(backlog_file, 'r') as f:
                content = f.read()
            
            # Extract next best value item (simplified parsing)
            lines = content.split('\n')
            next_item_section = False
            item_info = {}
            
            for line in lines:
                if "## 🎯 Next Best Value Item" in line:
                    next_item_section = True
                    continue
                elif next_item_section and line.startswith("**["):
                    # Parse item ID and title
                    item_line = line.strip("*").strip()
                    if "]" in item_line:
                        item_id = item_line.split("]")[0].strip("[")
                        item_title = item_line.split("]")[1].strip()
                        item_info["id"] = item_id
                        item_info["title"] = item_title
                elif next_item_section and line.startswith("- **"):
                    # Parse item details
                    if "Composite Score" in line:
                        score = line.split(":")[1].strip()
                        item_info["score"] = score
                    elif "Estimated Effort" in line:
                        effort = line.split(":")[1].strip()
                        item_info["effort"] = effort
                elif next_item_section and line.strip() == "":
                    break
            
            if item_info:
                print(f"🎯 Identified highest value item: {item_info.get('id', 'unknown')}")
                print(f"   Title: {item_info.get('title', 'unknown')}")
                print(f"   Score: {item_info.get('score', 'unknown')}")
                
                # For now, we'll log the execution intent
                # In a real implementation, this would execute the actual work
                execution_log = {
                    "timestamp": datetime.now().isoformat(),
                    "item_id": item_info.get("id"),
                    "item_title": item_info.get("title"),
                    "score": item_info.get("score"),
                    "status": "identified_for_execution",
                    "note": "Autonomous execution framework established"
                }
                
                self.state["last_execution"] = execution_log
                self._save_state()
                
                return {"success": True, "execution": execution_log}
            else:
                return {"success": False, "error": "No executable items found in backlog"}
                
        except Exception as e:
            return {"success": False, "error": f"Failed to parse backlog: {str(e)}"}
    
    def run_continuous_mode(self, duration_hours: int = 24):
        """Run in continuous mode for specified duration."""
        print(f"🚀 Starting continuous mode for {duration_hours} hours...")
        
        self.state["continuous_mode"] = True
        self.state["continuous_start"] = datetime.now().isoformat()
        self._save_state()
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        try:
            while datetime.now() < end_time and self.state.get("continuous_mode", False):
                # Check if discovery should run
                reason = self.should_run_discovery()
                
                if reason:
                    print(f"📊 Running discovery: {reason}")
                    discovery_result = self.run_value_discovery()
                    
                    if discovery_result["success"]:
                        print("✅ Discovery completed successfully")
                        
                        # Execute highest value item
                        execution_result = self.execute_highest_value_item()
                        if execution_result["success"]:
                            print("🎯 Highest value item identified for execution")
                        else:
                            print(f"⚠️  Execution identification failed: {execution_result.get('error')}")
                    else:
                        print(f"❌ Discovery failed: {discovery_result.get('error')}")
                
                # Sleep for 10 minutes before next check
                time.sleep(600)
                
        except KeyboardInterrupt:
            print("\n🛑 Continuous mode interrupted by user")
        finally:
            self.state["continuous_mode"] = False
            self.state["continuous_end"] = datetime.now().isoformat()
            self._save_state()
            print("🏁 Continuous mode ended")
    
    def run_single_cycle(self, force: bool = False):
        """Run a single discovery and execution cycle."""
        reason = self.should_run_discovery(force=force)
        
        if not reason and not force:
            print("⏰ No discovery needed at this time")
            return
        
        print(f"🔄 Running single cycle: {reason}")
        
        # Run discovery
        discovery_result = self.run_value_discovery()
        
        if discovery_result["success"]:
            print("✅ Discovery completed successfully")
            
            # Execute highest value item  
            execution_result = self.execute_highest_value_item()
            
            if execution_result["success"]:
                print("🎯 Execution cycle completed")
                return execution_result
            else:
                print(f"⚠️  Execution failed: {execution_result.get('error')}")
                return execution_result
        else:
            print(f"❌ Discovery failed: {discovery_result.get('error')}")
            return discovery_result
    
    def get_status(self) -> Dict:
        """Get current executor status."""
        return {
            "executor_state": self.state,
            "backlog_exists": (self.repo_path / "BACKLOG.md").exists(),
            "discovery_script_exists": self.discovery_script.exists(),
            "continuous_mode_active": self.state.get("continuous_mode", False),
            "last_discovery": self.state.get("last_discovery"),
            "execution_count": self.state.get("execution_count", 0)
        }


def main():
    """Main CLI interface for perpetual executor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terragon Perpetual SDLC Executor")
    parser.add_argument("--continuous", "-c", type=int, help="Run in continuous mode for N hours")
    parser.add_argument("--single", "-s", action="store_true", help="Run single discovery cycle")
    parser.add_argument("--force", "-f", action="store_true", help="Force execution regardless of schedule")
    parser.add_argument("--status", action="store_true", help="Show executor status")
    parser.add_argument("--setup-cron", action="store_true", help="Show cron setup instructions")
    
    args = parser.parse_args()
    
    executor = PerpetualExecutor()
    
    if args.status:
        status = executor.get_status()
        print("📊 TERRAGON EXECUTOR STATUS")
        print("=" * 50)
        print(f"Continuous Mode: {'🟢 Active' if status['continuous_mode_active'] else '🔴 Inactive'}")
        print(f"Executions: {status['execution_count']}")
        print(f"Last Discovery: {status['last_discovery'] or 'Never'}")
        print(f"Backlog Available: {'✅' if status['backlog_exists'] else '❌'}")
        
    elif args.setup_cron:
        print("📅 CRON SETUP INSTRUCTIONS")
        print("=" * 50)
        print("Add these lines to your crontab (crontab -e):")
        print()
        print("# Terragon Autonomous SDLC Scheduler")
        print(f"0 * * * *   cd {os.getcwd()} && python3 .terragon/perpetual_executor.py --single  # Hourly")
        print(f"0 2 * * *   cd {os.getcwd()} && python3 .terragon/perpetual_executor.py --single  # Daily at 2 AM")
        print(f"0 3 * * 1   cd {os.getcwd()} && python3 .terragon/perpetual_executor.py --single  # Weekly Monday 3 AM")
        print(f"0 4 1 * *   cd {os.getcwd()} && python3 .terragon/perpetual_executor.py --single  # Monthly 1st 4 AM")
        print()
        print("Or for continuous operation:")
        print(f"@reboot     cd {os.getcwd()} && python3 .terragon/perpetual_executor.py --continuous 168  # Run for 1 week")
        
    elif args.continuous:
        executor.run_continuous_mode(args.continuous)
        
    elif args.single:
        executor.run_single_cycle(force=args.force)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()