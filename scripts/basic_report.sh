#!/bin/bash
# Report metrics
echo "## Metrics"
git fetch --prune
dvc metrics diff master --show-md 

# Visual of logs
#echo "\`\`\`html\n" "$(cat siamese_logs.html)" "\n\`\`\`" >> reprort.md

# Visual of encoder model:
#echo "\`\`\`\n" "$(cat logs/distance_siamese_summaires/encoder.txt)" "\n\`\`\`" 

