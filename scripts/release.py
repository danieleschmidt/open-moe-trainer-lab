#!/usr/bin/env python3
"""
Automated Release Script for Open MoE Trainer Lab

This script automates the release process including:
- Version bumping
- Changelog generation
- Git tagging
- GitHub release creation
- Package publishing
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from packaging import version

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReleaseManager:
    """Manages the automated release process."""
    
    def __init__(self, repo_root: Optional[str] = None, dry_run: bool = False):
        self.repo_root = Path(repo_root or os.getcwd())
        self.dry_run = dry_run
        self.pyproject_path = self.repo_root / "pyproject.toml"
        self.changelog_path = self.repo_root / "CHANGELOG.md"
        
        if not self.pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found at {self.pyproject_path}")
    
    def get_current_version(self) -> str:
        """Get the current version from pyproject.toml."""
        with open(self.pyproject_path) as f:
            content = f.read()
        
        match = re.search(r'version = ["\']([^"\']+)["\']', content)
        if not match:
            raise ValueError("Could not find version in pyproject.toml")
        
        return match.group(1)
    
    def bump_version(self, current_version: str, bump_type: str) -> str:
        """Bump version according to semantic versioning."""
        v = version.parse(current_version)
        
        if bump_type == "patch":
            new_version = f"{v.major}.{v.minor}.{v.micro + 1}"
        elif bump_type == "minor":
            new_version = f"{v.major}.{v.minor + 1}.0"
        elif bump_type == "major":
            new_version = f"{v.major + 1}.0.0"
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
        
        return new_version
    
    def update_version_in_files(self, new_version: str) -> None:
        """Update version in pyproject.toml and other relevant files."""
        # Update pyproject.toml
        with open(self.pyproject_path) as f:
            content = f.read()
        
        updated_content = re.sub(
            r'version = ["\'][^"\']+["\']',
            f'version = "{new_version}"',
            content
        )
        
        if not self.dry_run:
            with open(self.pyproject_path, 'w') as f:
                f.write(updated_content)
        
        logger.info(f"Updated version to {new_version} in pyproject.toml")
    
    def get_commits_since_last_tag(self) -> List[Dict[str, str]]:
        """Get commits since the last git tag."""
        try:
            # Get the last tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True, text=True, check=True
            )
            last_tag = result.stdout.strip()
        except subprocess.CalledProcessError:
            # No previous tags
            last_tag = None
        
        # Get commits since last tag (or all commits if no tags)
        if last_tag:
            cmd = ["git", "log", f"{last_tag}..HEAD", "--pretty=format:%H|%s|%an|%ad", "--date=short"]
        else:
            cmd = ["git", "log", "--pretty=format:%H|%s|%an|%ad", "--date=short"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        commits = []
        for line in result.stdout.strip().split('\n'):
            if line:
                hash_val, subject, author, date = line.split('|', 3)
                commits.append({
                    'hash': hash_val,
                    'subject': subject,
                    'author': author,
                    'date': date
                })
        
        return commits
    
    def categorize_commits(self, commits: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Categorize commits by type (feat, fix, etc.)."""
        categories = {
            'features': [],
            'fixes': [],
            'docs': [],
            'style': [],
            'refactor': [],
            'performance': [],
            'tests': [],
            'build': [],
            'ci': [],
            'chore': [],
            'other': []
        }
        
        for commit in commits:
            subject = commit['subject'].lower()
            
            if subject.startswith(('feat:', 'feature:')):
                categories['features'].append(commit)
            elif subject.startswith('fix:'):
                categories['fixes'].append(commit)
            elif subject.startswith('docs:'):
                categories['docs'].append(commit)
            elif subject.startswith('style:'):
                categories['style'].append(commit)
            elif subject.startswith('refactor:'):
                categories['refactor'].append(commit)
            elif subject.startswith(('perf:', 'performance:')):
                categories['performance'].append(commit)
            elif subject.startswith('test:'):
                categories['tests'].append(commit)
            elif subject.startswith('build:'):
                categories['build'].append(commit)
            elif subject.startswith('ci:'):
                categories['ci'].append(commit)
            elif subject.startswith('chore:'):
                categories['chore'].append(commit)
            else:
                categories['other'].append(commit)
        
        return categories
    
    def generate_changelog_entry(self, version_str: str, commits: List[Dict[str, str]]) -> str:
        """Generate changelog entry for the new version."""
        categories = self.categorize_commits(commits)
        
        changelog = f"\n## [{version_str}] - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        if categories['features']:
            changelog += "### Added\n"
            for commit in categories['features']:
                changelog += f"- {commit['subject']} ({commit['hash'][:8]})\n"
            changelog += "\n"
        
        if categories['fixes']:
            changelog += "### Fixed\n"
            for commit in categories['fixes']:
                changelog += f"- {commit['subject']} ({commit['hash'][:8]})\n"
            changelog += "\n"
        
        if categories['performance']:
            changelog += "### Performance\n"
            for commit in categories['performance']:
                changelog += f"- {commit['subject']} ({commit['hash'][:8]})\n"
            changelog += "\n"
        
        if categories['docs']:
            changelog += "### Documentation\n"
            for commit in categories['docs']:
                changelog += f"- {commit['subject']} ({commit['hash'][:8]})\n"
            changelog += "\n"
        
        if categories['other']:
            changelog += "### Other Changes\n"
            for commit in categories['other']:
                changelog += f"- {commit['subject']} ({commit['hash'][:8]})\n"
            changelog += "\n"
        
        return changelog
    
    def update_changelog(self, version_str: str, commits: List[Dict[str, str]]) -> None:
        """Update CHANGELOG.md with new version entry."""
        new_entry = self.generate_changelog_entry(version_str, commits)
        
        if self.changelog_path.exists():
            with open(self.changelog_path) as f:
                existing_content = f.read()
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n"
        
        # Insert new entry after the header
        lines = existing_content.split('\n')
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('## [') or (i > 0 and not line.strip()):
                header_end = i
                break
        
        new_content = '\n'.join(lines[:header_end]) + new_entry + '\n'.join(lines[header_end:])
        
        if not self.dry_run:
            with open(self.changelog_path, 'w') as f:
                f.write(new_content)
        
        logger.info(f"Updated CHANGELOG.md with version {version_str}")
    
    def create_git_tag(self, version_str: str) -> None:
        """Create and push git tag for the new version."""
        tag_name = f"v{version_str}"
        
        if not self.dry_run:
            # Create annotated tag
            subprocess.run([
                "git", "tag", "-a", tag_name, 
                "-m", f"Release version {version_str}"
            ], check=True)
            
            # Push tag to origin
            subprocess.run(["git", "push", "origin", tag_name], check=True)
        
        logger.info(f"Created and pushed git tag {tag_name}")
    
    def trigger_github_release(self, version_str: str) -> None:
        """Trigger GitHub Actions release workflow."""
        if not self.dry_run:
            # This would typically use GitHub API or trigger workflow dispatch
            logger.info(f"Would trigger GitHub release workflow for v{version_str}")
        else:
            logger.info(f"[DRY RUN] Would trigger GitHub release workflow for v{version_str}")
    
    def release(self, bump_type: str, message: Optional[str] = None) -> None:
        """Execute the full release process."""
        logger.info(f"Starting {bump_type} release process...")
        
        # Get current version and calculate new version
        current_version = self.get_current_version()
        new_version = self.bump_version(current_version, bump_type)
        
        logger.info(f"Bumping version from {current_version} to {new_version}")
        
        # Get commits for changelog
        commits = self.get_commits_since_last_tag()
        logger.info(f"Found {len(commits)} commits since last release")
        
        if not self.dry_run:
            # Ensure we're on main branch and up to date
            subprocess.run(["git", "checkout", "main"], check=True)
            subprocess.run(["git", "pull", "origin", "main"], check=True)
            
            # Create release branch
            release_branch = f"release/v{new_version}"
            subprocess.run(["git", "checkout", "-b", release_branch], check=True)
        
        # Update version in files
        self.update_version_in_files(new_version)
        
        # Update changelog
        self.update_changelog(new_version, commits)
        
        if not self.dry_run:
            # Commit changes
            subprocess.run(["git", "add", "pyproject.toml", "CHANGELOG.md"], check=True)
            commit_msg = message or f"chore: release v{new_version}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            
            # Push release branch
            subprocess.run(["git", "push", "origin", release_branch], check=True)
            
            # Merge to main (would typically be done via PR)
            subprocess.run(["git", "checkout", "main"], check=True)
            subprocess.run(["git", "merge", release_branch], check=True)
            subprocess.run(["git", "push", "origin", "main"], check=True)
            
            # Clean up release branch
            subprocess.run(["git", "branch", "-d", release_branch], check=True)
            subprocess.run(["git", "push", "origin", "--delete", release_branch], check=True)
        
        # Create git tag
        self.create_git_tag(new_version)
        
        # Trigger GitHub release
        self.trigger_github_release(new_version)
        
        logger.info(f"Release v{new_version} completed successfully!")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Automated release script")
    parser.add_argument(
        "bump_type", 
        choices=["patch", "minor", "major"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--message", "-m",
        help="Custom commit message for release"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Perform a dry run without making changes"
    )
    parser.add_argument(
        "--repo-root",
        help="Path to repository root (default: current directory)"
    )
    
    args = parser.parse_args()
    
    try:
        release_manager = ReleaseManager(
            repo_root=args.repo_root,
            dry_run=args.dry_run
        )
        release_manager.release(args.bump_type, args.message)
    except Exception as e:
        logger.error(f"Release failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()