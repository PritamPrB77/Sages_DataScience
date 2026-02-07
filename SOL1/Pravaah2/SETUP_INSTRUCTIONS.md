# Causal RAG Analyzer - Setup Instructions

## Issue Fixed

The app was not responding to questions due to two issues:

1. ✅ **Query Blocking**: The app was blocking all queries when "All Outcomes" was selected (FIXED)
2. ❌ **Missing API Key**: The Google Gemini API key is not configured (NEEDS YOUR ACTION)

## Required: Configure Gemini API Key

The app uses Google's Gemini AI to perform causal analysis. You need to configure your API key.

### Step 1: Get Your Free API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

### Step 2: Set the Environment Variable

#### Option A: Temporary (Current Session Only)

**PowerShell:**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Command Prompt:**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

#### Option B: Permanent (Recommended)

**Windows:**
1. Press `Win + X` and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", click "New"
5. Variable name: `GEMINI_API_KEY`
6. Variable value: your-api-key-here
7. Click OK to save

**Or use PowerShell (run as Administrator):**
```powershell
[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your-api-key-here', 'User')
```

### Step 3: Restart Streamlit

After setting the environment variable:

1. Stop the current Streamlit server (Ctrl+C in terminal)
2. Restart it:
   ```powershell
   python -m streamlit run app.py
   ```

## How to Verify

1. After restarting, the main page should show: "✅ API key configured. Ready to load data!"
2. Load your dataset using the sidebar
3. Try asking a question - it should now work!

## Usage Tips

- **Select an Outcome**: For best results, select a specific outcome (like "Escalation") in the sidebar before asking questions
- **Quick Queries**: Use the quick query buttons for common questions
- **Custom Questions**: Type your own causal questions in the chat input
- **Evidence Panel**: Toggle "Show Retrieved Evidence" to see the source conversations

## Troubleshooting

### "API Key Missing" error persists

- Make sure you restarted Streamlit after setting the environment variable
- Verify the variable is set: `$env:GEMINI_API_KEY` (should show your key)
- If empty, try the permanent setup method above

### No response after asking questions

- Check that data is loaded (dataset statistics should appear in sidebar)
- Ensure you have internet connectivity (the API requires online access)
- Check the terminal/console for error messages

### Rate limit or quota errors

- The Gemini API has rate limits on the free tier
- Wait a few minutes between queries
- Consider upgrading to a paid plan for higher limits

## Additional Features

- **Reset Context**: Clear conversation history for a fresh start
- **Export Analysis**: Save your analysis results to JSON
- **Domain Filtering**: Filter conversations by business domain
- **Statistics Tab**: View dataset overview and outcome distribution

## Support

For issues or questions:
- Check the terminal logs for detailed error messages
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Review the API documentation: https://ai.google.dev/docs
