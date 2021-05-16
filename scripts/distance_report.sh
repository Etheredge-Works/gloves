# Report metrics
echo "## Metrics" >> report.md
git fetch --prune
dvc metrics diff master --show-md 

# Visual of logs
#echo "\`\`\`html\n" "$(cat siamese_logs.html)" "\n\`\`\`" >> reprort.md

# Visual of encoder model:
echo "\`\`\`\n" "$(cat logs/distance_siamese_logs/encoder.txt)" "\n\`\`\`" 
