# Contributing to Open MoE Trainer Lab

We welcome contributions from the community! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/open-moe-trainer-lab.git`
3. Set up the development environment: See [DEVELOPMENT.md](docs/DEVELOPMENT.md)
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## Types of Contributions

### Bug Reports

- Use the bug report template
- Provide detailed reproduction steps
- Include system information and logs
- Test on the latest version

### Feature Requests

- Use the feature request template
- Explain the use case and benefits
- Consider implementation complexity
- Discuss with maintainers first for large features

### Code Contributions

- Follow the [development guide](docs/DEVELOPMENT.md)
- Write tests for new functionality
- Update documentation
- Follow code style guidelines
- Sign your commits

### Documentation Improvements

- Fix typos and improve clarity
- Add examples and tutorials
- Update API documentation
- Translate documentation

## Development Process

### Setting Up Development Environment

See the [Development Guide](docs/DEVELOPMENT.md) for detailed setup instructions.

### Code Style

We use automated tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **pylint** for linting
- **mypy** for type checking

Run `pre-commit install` to set up automatic checks.

### Testing

- Write unit tests for new functions
- Add integration tests for features
- Ensure all tests pass: `npm test`
- Maintain test coverage above 80%

### Commit Guidelines

We follow [Conventional Commits](https://conventionalcommits.org/):

```
type(scope): description

Examples:
feat(routing): add expert choice routing algorithm
fix(training): resolve memory leak in expert computation
docs(api): update routing API documentation
test(integration): add distributed training tests
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Pull Request Process

1. **Before Creating PR:**
   - Ensure your branch is up to date with main
   - Run all tests and linting
   - Update documentation if needed
   - Add entry to CHANGELOG.md if significant

2. **PR Requirements:**
   - Use the PR template
   - Link related issues
   - Provide clear description of changes
   - Include screenshots for UI changes
   - Request review from relevant maintainers

3. **Review Process:**
   - At least one maintainer approval required
   - All CI checks must pass
   - Address review feedback promptly
   - Squash commits before merge if requested

## Contributing Guidelines

### New Features

1. **Discuss First**: For significant features, create an issue to discuss the approach
2. **Start Small**: Break large features into smaller, reviewable PRs
3. **Documentation**: Include comprehensive documentation and examples
4. **Tests**: Provide thorough test coverage
5. **Performance**: Consider performance implications

### Bug Fixes

1. **Reproduce**: Ensure you can reproduce the bug
2. **Root Cause**: Identify the underlying cause
3. **Test**: Add tests to prevent regression
4. **Minimal**: Keep changes focused on the specific issue

### Architecture Changes

1. **RFC Process**: Major architectural changes require an RFC (Request for Comments)
2. **Design Doc**: Create a design document for review
3. **Community Input**: Seek input from the community
4. **Migration Path**: Provide clear migration instructions

## Specific Contribution Areas

### Machine Learning Components

- **Models**: New MoE architectures and expert types
- **Routing**: Novel routing algorithms and load balancing
- **Training**: Optimization techniques and distributed training
- **Inference**: Performance optimizations and serving

### Infrastructure

- **Monitoring**: Metrics, dashboards, and alerting
- **Deployment**: Containerization and orchestration
- **CI/CD**: Build, test, and deployment automation
- **Security**: Vulnerability fixes and security features

### Documentation

- **Tutorials**: Step-by-step guides for common use cases
- **API Docs**: Comprehensive API reference
- **Examples**: Real-world usage examples
- **Translations**: Documentation in other languages

### Community

- **Issue Triage**: Help triage and label issues
- **Support**: Answer questions in discussions
- **Mentoring**: Help new contributors get started
- **Events**: Organize community events and workshops

## Recognition

We value all contributions and recognize contributors in several ways:

- **Contributors List**: All contributors are listed in the repository
- **Release Notes**: Significant contributions are highlighted
- **Swag**: Active contributors receive project swag
- **Speaking Opportunities**: Contributors may be invited to present

## Legal Requirements

### Contributor License Agreement (CLA)

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (Apache 2.0).

### Sign-off Requirement

All commits must be signed off to certify that you have the right to submit the code:

```bash
git commit -s -m "Your commit message"
```

This adds a "Signed-off-by" line to your commit message.

## Getting Help

- **Documentation**: Check existing docs first
- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Create issues for bugs or feature requests
- **Discord**: Join our community Discord server
- **Email**: Contact maintainers directly for sensitive issues

## Maintainer Guidelines

### For Maintainers

- **Responsive**: Respond to PRs and issues within 48 hours
- **Constructive**: Provide helpful feedback and suggestions
- **Inclusive**: Welcome contributions from all backgrounds
- **Quality**: Maintain high standards while being supportive

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No breaking changes without major version bump
- [ ] Performance implications considered
- [ ] Security implications reviewed

## Release Process

1. **Feature Freeze**: No new features for upcoming release
2. **Testing**: Comprehensive testing on multiple platforms
3. **Documentation**: Update all documentation
4. **Release Notes**: Prepare detailed release notes
5. **Tagging**: Create release tag and publish
6. **Announcement**: Announce release to community

## Thank You!

Thank you for contributing to Open MoE Trainer Lab! Your contributions help make this project better for everyone.
