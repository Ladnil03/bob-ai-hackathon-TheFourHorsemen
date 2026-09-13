# Problem Statement

## Background
Advanced chip manufacturing at 3nm and 5nm nodes is incredibly sensitive. 
Wafers cost $100K+ each. A single batch of 25 wafers processes through 
14+ steps over 2–3 weeks. If yield drops, engineers must find why.

## The Problem
When yield is low, finding the root cause takes 1–2 weeks of manual analysis:
- correlating ~500 equipment sensor readings per batch
- mapping defect locations to process parameters
- comparing against 100+ historical wafers manually

Cost: $50–100M/month per 1% yield loss. Every day of delay is millions lost.

## Who Is Affected
Process engineers at 3nm/5nm fabs who own manufacturing quality.
Equipment engineers who tune process recipes.
Manufacturing managers tracking fab performance.

## Why It Matters
Reactive root cause analysis is too slow. Manufacturers need to:
1. Predict which upcoming batches will fail (before they run)
2. Identify root causes in hours, not weeks
3. Reduce MTTR (mean time to recovery)

## Why Existing Solutions Don't Work
- Manual analysis: Too slow, labor-intensive
- Heuristic rules: Can't capture complex multi-dimensional interactions
- Generic ML: Doesn't explain WHY (black box)
- Equipment vendors' dashboards: Show raw data, not insights
