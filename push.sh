# push.sh
git add .
git commit -m "AI update: $(date +%H:%M:%S)"
git push origin main # 或者是你的分支名
echo "✅ 已推送到云端"