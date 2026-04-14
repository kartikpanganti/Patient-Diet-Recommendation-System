# Patient Diet Recommendation System (Streamlit + Gemini)

Visite Site:
https://patient-diet-recommendation-system.streamlit.app

A Streamlit web app that collects basic patient details (age, gender, height/weight, diet preference, disease, location, allergies, food type) and generates:

- Restaurant recommendations
- Breakfast options
- Dinner options
- Workout suggestions

Generation is powered by Google Gemini via LangChain.

## Features (what the app does)

- **Single-page Streamlit UI** with a patient input form
- **Prompt-driven recommendations** using a structured prompt template
- **Gemini model call** via `langchain-google-genai`
- **Works on both old/new LangChain versions** (falls back if `LLMChain` is unavailable)
- **Optional recipe recommendations** using (optional) ingredients + time inputs

## End-to-end workflow

1. User opens the Streamlit app.
2. User fills in all fields:
	- age, gender, weight, height
	- veg/non-veg preference
	- disease
	- region and state/city
	- allergies
	- food type
	- (optional) recipe inputs: ingredients available, ingredients to avoid, and max cooking time
3. User clicks **Get Recommendations**.
4. The app:
	- builds an `input_data` dictionary from the form
	- formats the prompt using the template
	- calls Gemini (LLM) through LangChain
	- prints the generated text to the page

## Technical overview

### Tech stack

- **Frontend/UI**: Streamlit
- **LLM orchestration**: LangChain (`langchain`, `langchain-core`)
- **Gemini integration**: `langchain-google-genai`
- **Provider SDK**: `google-generativeai`

### Project structure

```
.
├─ app.py               # Streamlit app
├─ requirements.txt     # Python dependencies
├─ README.md            # Documentation
└─ .env                 # (local only) environment variables (DO NOT COMMIT)
```

### How the prompt is built

The app uses a `PromptTemplate` with these input variables:

`age, gender, weight, height, veg_or_nonveg, disease, region, state, allergics, foodtype`

The prompt asks the model to recommend:

- 6 restaurant names
- 6 breakfast names
- 5 dinner names
- 6 workout names

### How the model is called

In `app.py`, the Gemini LLM is created with:

- `model="gemini-1.5-pro"`
- `generation_config` such as `temperature` and `max_output_tokens`

Then one of these paths is used depending on the installed LangChain version:

- **Older LangChain**: `LLMChain(llm=model, prompt=prompt_template)`
- **Newer LangChain**: `prompt_template | model` (Runnable pipeline)

## Configuration

### Required: `GOOGLE_API_KEY`

The app requires a Google API key in either:

- Environment variable: `GOOGLE_API_KEY`
- Streamlit Cloud Secrets: `st.secrets["GOOGLE_API_KEY"]`

If the key is missing, the app stops early and shows a clear error message.

### Local `.env` note

This repo includes a `.env` file, but **Python will not automatically load it** unless you load it yourself (via your shell, an IDE setting, or by adding a dotenv loader).

Recommended local options:

1. PowerShell (temporary for the current terminal):
	- `$env:GOOGLE_API_KEY="your_key_here"`
2. Windows (persist for future terminals):
	- `setx GOOGLE_API_KEY "your_key_here"`
	  - then reopen your terminal

Do **not** commit real API keys to Git.

## Run locally (no venv)

These commands install dependencies globally for your current Python.

1. Install dependencies:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

2. Set API key:

```powershell
$env:GOOGLE_API_KEY = "your_key_here"
```

3. Run Streamlit:

```powershell
py -m streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Create a new Streamlit Cloud app pointing to this repository.
3. In Streamlit Cloud:
	- Open **App → Settings → Secrets**
	- Add:
	  - `GOOGLE_API_KEY = "your_key_here"`
4. Rerun/redeploy.

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain'`

Usually means dependency installation failed.

- Confirm `requirements.txt` is present and correct.
- On Streamlit Cloud, check build logs for a pip error.

### `ModuleNotFoundError` for `langchain_google_genai`

The **pip package name uses hyphens**:

- Correct: `langchain-google-genai`
- Incorrect: `langchain_google_genai`

### `from langchain.chains import LLMChain` fails

Some LangChain versions remove/relocate `LLMChain`. The app is written to work even when it’s missing by using the runnable pipeline: `prompt | model`.

### `ChatGoogleGenerativeAIError` (Gemini call fails)

This is an API/provider error. Common causes:

- **Invalid or missing API key**
- **Quota exceeded / billing disabled**
- **Model access not enabled** for your project
- **Unsupported model name** in your region/account

Fix checklist:

1. Re-check `GOOGLE_API_KEY` (Secrets on Streamlit Cloud, env var locally).
2. Verify the key works in Google AI Studio / Google Cloud.
3. If needed, change the model name in `app.py` to one your key can access.

## Security notes

- Never commit `.env` containing API keys.
- Use Streamlit Secrets for deployed apps.

## Credits

2024 CODE_WIZARDS

- @kartik panaganti
- @Zahid shaikh
- @vijaykumar Maske
