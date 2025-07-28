#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generator for Open MoE Trainer Lab

Generates SPDX and CycloneDX format SBOMs for the project.
"""

import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pkg_resources
import toml


class SBOMGenerator:
    """Generate Software Bill of Materials for the project."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.package_json_path = self.project_root / "package.json"
        
    def get_project_info(self) -> Dict[str, Any]:
        """Extract project information from pyproject.toml."""
        with open(self.pyproject_path, 'r') as f:
            pyproject = toml.load(f)
            
        project = pyproject.get('project', {})
        return {
            'name': project.get('name', 'open-moe-trainer-lab'),
            'version': project.get('version', '0.1.0'),
            'description': project.get('description', ''),
            'license': project.get('license', {}).get('file', 'LICENSE'),
            'authors': project.get('authors', []),
            'homepage': project.get('urls', {}).get('Homepage', ''),
            'repository': project.get('urls', {}).get('Repository', '')
        }
    
    def get_python_dependencies(self) -> List[Dict[str, Any]]:
        """Get Python package dependencies."""
        dependencies = []
        
        try:
            # Get installed packages
            installed_packages = [d for d in pkg_resources.working_set]
            
            for package in sorted(installed_packages, key=lambda x: x.project_name):
                dep_info = {
                    'name': package.project_name,
                    'version': package.version,
                    'type': 'python-package',
                    'source': 'pypi',
                    'license': self._get_package_license(package),
                    'location': package.location if hasattr(package, 'location') else None
                }
                dependencies.append(dep_info)
                
        except Exception as e:
            print(f"Warning: Could not get Python dependencies: {e}")
            
        return dependencies
    
    def get_node_dependencies(self) -> List[Dict[str, Any]]:
        """Get Node.js package dependencies."""
        dependencies = []
        
        if not self.package_json_path.exists():
            return dependencies
            
        try:
            # Run npm list to get dependency tree
            result = subprocess.run(
                ['npm', 'list', '--json', '--all'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                npm_data = json.loads(result.stdout)
                dependencies = self._parse_npm_dependencies(npm_data.get('dependencies', {}))
                
        except Exception as e:
            print(f"Warning: Could not get Node.js dependencies: {e}")
            
        return dependencies
    
    def _parse_npm_dependencies(self, deps: Dict[str, Any], prefix: str = '') -> List[Dict[str, Any]]:
        """Parse npm dependency tree recursively."""
        dependencies = []
        
        for name, info in deps.items():
            dep_info = {
                'name': name,
                'version': info.get('version', 'unknown'),
                'type': 'npm-package',
                'source': 'npm',
                'license': info.get('license'),
                'location': info.get('path')
            }
            dependencies.append(dep_info)
            
            # Parse nested dependencies
            if 'dependencies' in info:
                nested_deps = self._parse_npm_dependencies(
                    info['dependencies'], 
                    f"{prefix}{name}/"
                )
                dependencies.extend(nested_deps)
                
        return dependencies
    
    def _get_package_license(self, package) -> str:
        """Attempt to get license information for a Python package."""
        try:
            # Try to get license from metadata
            if hasattr(package, 'get_metadata'):
                metadata = package.get_metadata('METADATA')
                for line in metadata.split('\n'):
                    if line.startswith('License:'):
                        return line.split(':', 1)[1].strip()
            return 'Unknown'
        except Exception:
            return 'Unknown'
    
    def get_system_dependencies(self) -> List[Dict[str, Any]]:
        """Get system-level dependencies."""
        dependencies = []
        
        # Docker base image info
        try:
            dockerfile_path = self.project_root / "Dockerfile"
            if dockerfile_path.exists():
                with open(dockerfile_path, 'r') as f:
                    content = f.read()
                    
                # Extract base images
                for line in content.split('\n'):
                    if line.strip().startswith('FROM '):
                        image = line.split('FROM')[1].split('as')[0].strip()
                        dependencies.append({
                            'name': image.split(':')[0],
                            'version': image.split(':')[1] if ':' in image else 'latest',
                            'type': 'container-image',
                            'source': 'docker-hub'
                        })
        except Exception as e:
            print(f"Warning: Could not parse Dockerfile: {e}")
            
        return dependencies
    
    def generate_spdx_sbom(self) -> Dict[str, Any]:
        """Generate SPDX format SBOM."""
        project_info = self.get_project_info()
        python_deps = self.get_python_dependencies()
        node_deps = self.get_node_dependencies()
        system_deps = self.get_system_dependencies()
        
        all_deps = python_deps + node_deps + system_deps
        
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"{project_info['name']}-sbom",
            "documentNamespace": f"https://github.com/your-org/{project_info['name']}/sbom/{datetime.now().isoformat()}",
            "creationInfo": {
                "created": datetime.now().isoformat(),
                "creators": ["Tool: sbom-generator"],
                "licenseListVersion": "3.19"
            },
            "packages": [
                {
                    "SPDXID": "SPDXRef-Package",
                    "name": project_info['name'],
                    "downloadLocation": project_info.get('repository', 'NOASSERTION'),
                    "filesAnalyzed": False,
                    "licenseConcluded": "NOASSERTION",
                    "licenseDeclared": "NOASSERTION",
                    "copyrightText": "NOASSERTION",
                    "versionInfo": project_info['version']
                }
            ],
            "relationships": []
        }
        
        # Add dependencies as packages
        for i, dep in enumerate(all_deps):
            package_id = f"SPDXRef-Package-{i+1}"
            sbom["packages"].append({
                "SPDXID": package_id,
                "name": dep['name'],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": dep.get('license', 'NOASSERTION'),
                "licenseDeclared": dep.get('license', 'NOASSERTION'),
                "copyrightText": "NOASSERTION",
                "versionInfo": dep['version']
            })
            
            # Add relationship
            sbom["relationships"].append({
                "spdxElementId": "SPDXRef-Package",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": package_id
            })
            
        return sbom
    
    def generate_cyclonedx_sbom(self) -> Dict[str, Any]:
        """Generate CycloneDX format SBOM."""
        project_info = self.get_project_info()
        python_deps = self.get_python_dependencies()
        node_deps = self.get_node_dependencies()
        system_deps = self.get_system_dependencies()
        
        all_deps = python_deps + node_deps + system_deps
        
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tools": [
                    {
                        "vendor": "Open MoE Trainer Lab",
                        "name": "sbom-generator",
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "application",
                    "bom-ref": "pkg:pypi/open-moe-trainer-lab",
                    "name": project_info['name'],
                    "version": project_info['version'],
                    "description": project_info['description']
                }
            },
            "components": []
        }
        
        # Add dependencies as components
        for dep in all_deps:
            component = {
                "type": "library",
                "bom-ref": f"pkg:{dep['type']}/{dep['name']}@{dep['version']}",
                "name": dep['name'],
                "version": dep['version'],
                "scope": "required"
            }
            
            if dep.get('license'):
                component["licenses"] = [
                    {"license": {"name": dep['license']}}
                ]
                
            sbom["components"].append(component)
            
        return sbom
    
    def save_sbom(self, sbom: Dict[str, Any], filename: str) -> None:
        """Save SBOM to file."""
        output_path = self.project_root / "security" / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(sbom, f, indent=2)
            
        print(f"SBOM saved to {output_path}")


def main():
    """Main function to generate SBOMs."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generator = SBOMGenerator(project_root)
    
    print("Generating Software Bill of Materials...")
    
    # Generate SPDX SBOM
    print("Generating SPDX SBOM...")
    spdx_sbom = generator.generate_spdx_sbom()
    generator.save_sbom(spdx_sbom, "sbom-spdx.json")
    
    # Generate CycloneDX SBOM
    print("Generating CycloneDX SBOM...")
    cyclonedx_sbom = generator.generate_cyclonedx_sbom()
    generator.save_sbom(cyclonedx_sbom, "sbom-cyclonedx.json")
    
    print("SBOM generation complete!")
    print(f"Total components: {len(spdx_sbom['packages']) - 1}")


if __name__ == "__main__":
    main()