@echo off
cd /d "D:\Tekworks\AI-Powered Credit Card Fraud Detection System\AI-Powered-Credit-Card-Fraud-Detection-System"

echo === Checking git status ===
git status

echo.
echo === Checking remotes ===
git remote -v

echo.
echo === Adding remote if not exists ===
git remote remove origin 2>nul
git remote add origin https://github.com/AnveshAnnepaga/AI-Powered-Credit-Card-Fraud-Detection-System.git

echo.
echo === Staging all files ===
git add .

echo.
echo === Committing ===
git commit -m "Initial commit: AI-Powered Credit Card Fraud Detection System"

echo.
echo === Setting branch to main ===
git branch -M main

echo.
echo === Pushing to GitHub ===
git push -u origin main

echo.
echo === Done! ===
pause
