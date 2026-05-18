#!/bin/bash
# Release script for Agent Lightning
set -e

VERSION=${1:-"0.3.0"}
echo "⚡ Releasing Agent Lightning v${VERSION}"

# Bump version
sed -i "s/__version__ = \".*\"/__version__ = \"${VERSION}\"/" agent_lightning/__init__.py
sed -i "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml

# Run tests
echo "Running tests..."
python -m pytest tests/ -v --tb=short

# Build
echo "Building..."
python -m build

# Tag
git add -A
git commit -m "chore: release v${VERSION}"
git tag -a "v${VERSION}" -m "Release v${VERSION}"

echo "✅ v${VERSION} ready. Push with: git push && git push --tags"
echo "   Publish with: twine upload dist/*"
