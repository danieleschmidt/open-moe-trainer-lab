# Advanced Security Workflows

This document provides GitHub Actions workflow templates for advanced security scanning and compliance monitoring.

> **Note**: Due to GitHub permissions, these workflows must be manually added to the `.github/workflows/` directory.

## 🔍 CodeQL Advanced Security Analysis

Create `.github/workflows/codeql.yml`:

```yaml
name: "CodeQL Advanced Security Analysis"

on:
  push:
    branches: [ "main", "develop" ]
  pull_request:
    branches: [ "main" ]
  schedule:
    - cron: '45 6 * * 2'

jobs:
  analyze:
    name: Analyze (${{ matrix.language }})
    runs-on: ${{ (matrix.language == 'swift' && 'macos-latest') || 'ubuntu-latest' }}
    timeout-minutes: ${{ (matrix.language == 'swift' && 120) || 360 }}
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python', 'javascript-typescript' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}
        queries: security-extended,security-and-quality

    - name: Autobuild
      uses: github/codeql-action/autobuild@v3

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
      with:
        category: "/language:${{matrix.language}}"
```

## 🔒 Dependency Security Review

Create `.github/workflows/dependency-review.yml`:

```yaml
name: 'Dependency Review'
on: [pull_request]

permissions:
  contents: read

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    steps:
      - name: 'Checkout Repository'
        uses: actions/checkout@v4
      - name: 'Dependency Review'
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: moderate
          allow-licenses: MIT, Apache-2.0, BSD-3-Clause, BSD-2-Clause, ISC, CC0-1.0
          deny-licenses: GPL-3.0, LGPL-3.0, AGPL-3.0
```

## 🛡️ Advanced Security Scanning

Create `.github/workflows/security-scan.yml`:

```yaml
name: "Advanced Security Scanning"

on:
  push:
    branches: [ "main", "develop" ]
  pull_request:
    branches: [ "main" ]
  schedule:
    - cron: '0 8 * * 1'  # Weekly on Monday

jobs:
  security-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          pip install safety bandit semgrep
      
      - name: Run Bandit security scan
        run: |
          bandit -r . -f json -o bandit-report.json || true
          bandit -r . -f txt
      
      - name: Run Safety dependency scan
        run: |
          safety check --json --output safety-report.json || true
          safety check
      
      - name: Run Semgrep security scan
        run: |
          semgrep --config=auto --json --output=semgrep-report.json . || true
          semgrep --config=auto .
      
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json
            semgrep-report.json
```

## 🔐 Container Security Scanning

Create `.github/workflows/container-security.yml`:

```yaml
name: "Container Security Scanning"

on:
  push:
    branches: [ "main", "develop" ]
    paths: 
      - 'Dockerfile'
      - 'docker-compose*.yml'
  pull_request:
    branches: [ "main" ]
    paths:
      - 'Dockerfile'
      - 'docker-compose*.yml'

jobs:
  container-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Build Docker image
        run: |
          docker build -t moe-trainer:security-scan .
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'moe-trainer:security-scan'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Run Hadolint Dockerfile linter
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          format: sarif
          output-file: hadolint-results.sarif
          no-fail: true
      
      - name: Upload Hadolint scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: hadolint-results.sarif
```

## 📊 Compliance Monitoring

Create `.github/workflows/compliance-check.yml`:

```yaml
name: "Security Compliance Check"

on:
  push:
    branches: [ "main" ]
  schedule:
    - cron: '0 9 * * 1'  # Weekly on Monday at 9 AM
  workflow_dispatch:  # Manual trigger

jobs:
  compliance-check:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          pip install cyclonedx-bom pyyaml
      
      - name: Run compliance report
        run: |
          python security/compliance-report.py --output compliance-report.json
      
      - name: Upload compliance report
        uses: actions/upload-artifact@v3
        with:
          name: compliance-report
          path: compliance-report.json
      
      - name: Check compliance score
        run: |
          python -c "
          import json
          with open('compliance-report.json') as f:
              report = json.load(f)
          score = report['summary']['compliance_score']
          print(f'Compliance Score: {score}%')
          if score < 80:
              print('::error::Compliance score below threshold (80%)')
              exit(1)
          "
      
      - name: Create issue on compliance failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'Security Compliance Check Failed',
              body: `
                ## 🚨 Security Compliance Check Failed
                
                The automated compliance check has failed. Please review the compliance report and address any issues.
                
                **Workflow Run**: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
                
                **Next Steps**:
                1. Download the compliance report artifact
                2. Review failed compliance checks
                3. Address security issues
                4. Re-run the compliance check
                
                @security-team please review.
              `,
              labels: ['security', 'compliance', 'urgent']
            })
```

## 🔄 SBOM Generation

Create `.github/workflows/sbom-generation.yml`:

```yaml
name: "SBOM Generation"

on:
  release:
    types: [published]
  push:
    branches: [ "main" ]
  workflow_dispatch:

jobs:
  generate-sbom:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[all]"
          pip install cyclonedx-bom
      
      - name: Generate SBOM
        run: |
          cyclonedx-bom -o sbom.json
          cyclonedx-bom -o sbom.xml --format xml
      
      - name: Verify SBOM
        run: |
          python -c "
          import json
          with open('sbom.json') as f:
              sbom = json.load(f)
          components = len(sbom.get('components', []))
          print(f'SBOM contains {components} components')
          if components < 10:
              print('::warning::SBOM seems incomplete')
          "
      
      - name: Upload SBOM artifacts
        uses: actions/upload-artifact@v3
        with:
          name: sbom-files
          path: |
            sbom.json
            sbom.xml
      
      - name: Attach SBOM to release
        if: github.event_name == 'release'
        uses: softprops/action-gh-release@v1
        with:
          files: |
            sbom.json
            sbom.xml
```

## 🚀 Setup Instructions

1. **Create the workflow files** in your `.github/workflows/` directory
2. **Configure repository secrets** for any external services
3. **Enable security features** in your repository settings:
   - Code scanning alerts
   - Dependency alerts
   - Secret scanning
4. **Set up branch protection** rules to require security checks
5. **Configure notification settings** for security alerts

## 🔧 Customization

### Environment Variables
```yaml
env:
  SECURITY_SCAN_LEVEL: "high"
  COMPLIANCE_THRESHOLD: "80"
  NOTIFICATION_WEBHOOK: ${{ secrets.SECURITY_WEBHOOK }}
```

### Custom Security Rules
```yaml
- name: Custom security checks
  run: |
    python scripts/custom-security-check.py
    python security/compliance-report.py --threshold 85
```

### Integration with External Tools
```yaml
- name: Upload to security platform
  env:
    SECURITY_PLATFORM_TOKEN: ${{ secrets.SECURITY_TOKEN }}
  run: |
    curl -X POST -H "Authorization: Bearer $SECURITY_PLATFORM_TOKEN" \
         -F "file=@compliance-report.json" \
         https://security-platform.example.com/api/reports
```

---

These workflows provide comprehensive security coverage for advanced ML/AI repositories with automatic scanning, compliance monitoring, and vulnerability management.