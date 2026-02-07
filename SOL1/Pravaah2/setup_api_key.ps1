# Causal RAG Analyzer - API Key Setup Script
# This script helps you configure your Google Gemini API key

Write-Host "`n=== Causal RAG Analyzer - API Key Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if API key is already set
$currentKey = $env:GEMINI_API_KEY
if ($currentKey) {
    Write-Host "✓ API key is already set (first 10 chars): $($currentKey.Substring(0, [Math]::Min(10, $currentKey.Length)))..." -ForegroundColor Green
    $response = Read-Host "`nDo you want to update it? (y/n)"
    if ($response -ne 'y') {
        Write-Host "`nNo changes made. You can start the app now." -ForegroundColor Yellow
        exit
    }
}
else {
    Write-Host "⚠ No API key is currently set" -ForegroundColor Yellow
}

Write-Host "`n--- How to Get Your API Key ---" -ForegroundColor White
Write-Host "1. Visit: https://makersuite.google.com/app/apikey"
Write-Host "2. Sign in with your Google account"
Write-Host "3. Click 'Create API Key' or 'Get API Key'"
Write-Host "4. Copy the generated key"
Write-Host ""

# Prompt for API key
$apiKey = Read-Host "Enter your Gemini API key (or press Enter to cancel)"

if (-not $apiKey) {
    Write-Host "`nSetup cancelled." -ForegroundColor Yellow
    exit
}

# Validate the key format (basic check)
if ($apiKey.Length -lt 20) {
    Write-Host "`n⚠ Warning: The API key seems too short. Please verify it's correct." -ForegroundColor Yellow
}

# Ask for setup type
Write-Host "`n--- Setup Type ---" -ForegroundColor White
Write-Host "1. Temporary (current session only)"
Write-Host "2. Permanent (for your user account)"
Write-Host ""
$setupType = Read-Host "Choose setup type (1 or 2)"

if ($setupType -eq "2") {
    # Permanent setup
    try {
        [System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', $apiKey, 'User')
        Write-Host "`n✓ API key saved permanently!" -ForegroundColor Green
        Write-Host "  The key will be available in all new PowerShell sessions." -ForegroundColor Gray
    }
    catch {
        Write-Host "`n✗ Failed to save permanently: $_" -ForegroundColor Red
        Write-Host "  Falling back to temporary setup..." -ForegroundColor Yellow
        $setupType = "1"
    }
}

if ($setupType -eq "1") {
    # Temporary setup
    $env:GEMINI_API_KEY = $apiKey
    Write-Host "`n✓ API key set for current session!" -ForegroundColor Green
    Write-Host "  Note: You'll need to set it again after closing this terminal." -ForegroundColor Gray
}

# Verify it's set
$verifyKey = $env:GEMINI_API_KEY
if ($verifyKey) {
    Write-Host "`n✓ Verification successful!" -ForegroundColor Green
    Write-Host "  Key starts with: $($verifyKey.Substring(0, [Math]::Min(15, $verifyKey.Length)))..." -ForegroundColor Gray
}
else {
    Write-Host "`n✗ Verification failed. Please try again." -ForegroundColor Red
    exit
}

# Instructions to start the app
Write-Host "`n--- Next Steps ---" -ForegroundColor Cyan
Write-Host "1. Run the app: python -m streamlit run app.py"
Write-Host "2. Load your dataset using the sidebar"
Write-Host "3. Select a specific outcome (or keep 'All outcomes')"
Write-Host "4. Start asking questions!"
Write-Host ""
Write-Host "Setup complete! 🎉" -ForegroundColor Green
Write-Host ""
