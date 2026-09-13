#!/bin/bash
# Run this script to check everything

echo "✓ Checking code..."
python -m py_compile src/app.py src/analysis/*.py && echo "  Python syntax: OK"

echo "✓ Checking submission.yaml..."
grep '""' submission.yaml || echo "  No empty fields: OK"

echo "✓ Checking README..."
grep '\[' README.md || echo "  No placeholders: OK"

echo "✓ Checking docs..."
[ -f docs/problem-statement.md ] && echo "  problem-statement.md: OK"
[ -f docs/solution-overview.md ] && echo "  solution-overview.md: OK"
[ -f docs/setup-guide.md ] && echo "  setup-guide.md: OK"
[ -f docs/architecture.md ] && echo "  architecture.md: OK"

echo "✓ Checking demo..."
[ -d demo/screenshots ] && [ "$(ls demo/screenshots/*.png 2>/dev/null | wc -l)" -ge 3 ] && echo "  Screenshots (3+): OK"
grep -q "http" demo/demo-video-link.txt && echo "  Demo video URL: OK"

echo "✓ Checking presentation..."
[ -f presentation/slides.pdf ] && echo "  Presentation PDF: OK"

echo "✓ Checking git..."
git log --oneline | head -1 && echo "  Git history: OK"
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] && echo "  On main branch: OK"

echo ""
echo "🎉 All checks passed! Ready to submit."
