@echo off
echo ===============================================================
echo   Automated Plant Leaf Disease Detection - GitHub Push Helper
echo ===============================================================
echo.
set /p REPO_URL="Paste your GitHub Repository URL (e.g., https://github.com/your-username/your-repo.git): "
if "%REPO_URL%"=="" (
    echo [ERROR] No URL entered. Aborting.
    pause
    exit /b 1
)

echo.
echo [+] Configuring remote origin...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo [+] Setting default branch to main...
git branch -M main

echo [+] Staging and committing any uncommitted changes...
git add .
git commit -m "update: plant leaf disease detection & severity estimation codebase" 2>nul

echo.
echo [+] Pushing repository to GitHub (%REPO_URL%)...
git push -u origin main

echo.
if %errorlevel% equ 0 (
    echo [SUCCESS] Your project has been pushed to GitHub successfully!
) else (
    echo [ERROR] Push failed. Please check your GitHub credentials or repo URL.
)
echo.
pause
