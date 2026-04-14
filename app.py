import os
from pathlib import Path
import streamlit as st
import re
try:
    from langchain_core.prompts import PromptTemplate
except Exception:  # pragma: no cover
    from langchain.prompts import PromptTemplate

from langchain_google_genai import GoogleGenerativeAI

try:
    from langchain.chains import LLMChain
except Exception:  # pragma: no cover
    LLMChain = None

try:
    import langchain.globals as lcg
except Exception:  # pragma: no cover
    lcg = None

# Set verbose to True or False based on your requirements
if lcg is not None:
    lcg.set_verbose(True)  # Enable verbose mode if needed

# Model setup
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        google_api_key = None

gemini_model = os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL")
if not gemini_model:
    try:
        gemini_model = st.secrets.get("GEMINI_MODEL")  # type: ignore[attr-defined]
    except Exception:
        gemini_model = None

if not gemini_model:
    # Default to a commonly available generateContent-capable model.
    # Override via env var `GEMINI_MODEL` (or Streamlit Secrets) if your project/account differs.
    gemini_model = "gemini-3.1-flash-lite-preview"

if not google_api_key:
    st.error(
        "Missing GOOGLE_API_KEY. Set it as an environment variable or in Streamlit Secrets."
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key
model = GoogleGenerativeAI(
    model=gemini_model,
    temperature=0.7,
    top_p=1,
    top_k=1,
    max_output_tokens=2048,
)
# promt template
prompt_template_resto = PromptTemplate(
    input_variables=[
        'age',
        'gender',
        'weight',
        'height',
        'veg_or_nonveg',
        'disease',
        'region',
        'state',
        'allergics',
        'foodtype',
        'ingredients_have',
        'ingredients_avoid',
        'max_cook_time_minutes',
    ],
    template="Diet Recommendation System:\n"
             "You must follow the HARD CONSTRAINTS below. If you cannot satisfy a constraint, do not suggest that item.\n\n"
             "HARD CONSTRAINTS:\n"
             "- Veg vs Non-Veg is STRICT. If Veg, do NOT include any meat, fish/seafood, or eggs.\n"
             "- Do NOT include allergens listed in 'Person allergics'.\n"
             "- Do NOT include ingredients listed in 'Ingredients to avoid'.\n"
             "- Respect disease constraints when choosing foods (e.g., diabetes, BP, etc.).\n\n"
             "I want you to recommend 6 restaurants names, 6 breakfast names, 5 dinner names, and 6 workout names, "
             "based on the following criteria:\n"
             "Person age: {age}\n"
             "Person gender: {gender}\n"
             "Person weight: {weight}\n"
             "Person height: {height}\n"
             "Person veg_or_nonveg: {veg_or_nonveg}\n"
             "Person generic disease: {disease}\n"
             "Person region: {region}\n"
             "Person state or City: {state}\n"  
             "Person allergics: {allergics}\n"
             "Person foodtype: {foodtype}.\n\n"
             "Optional recipe recommendation inputs (may be empty):\n"
             "Ingredients available: {ingredients_have}\n"
             "Ingredients to avoid: {ingredients_avoid}\n"
             "Max cooking time (minutes): {max_cook_time_minutes}\n\n"
             "Also recommend 3 simple recipes that match the person’s diet preferences and disease constraints. "
             "If ingredients are provided, prefer using available ingredients; if exclusions are provided, do not use them. "
             "If max cooking time is provided, keep each recipe within that time. "
             "For each recipe, include: name, estimated time, key ingredients, and short steps."
)

if LLMChain is not None:
    chain_resto = LLMChain(llm=model, prompt=prompt_template_resto)
else:
    chain_resto = prompt_template_resto | model


prompt_template_repair = PromptTemplate(
    input_variables=[
        'age',
        'gender',
        'weight',
        'height',
        'veg_or_nonveg',
        'disease',
        'region',
        'state',
        'allergics',
        'foodtype',
        'ingredients_have',
        'ingredients_avoid',
        'max_cook_time_minutes',
        'original_text',
    ],
    template=(
        "You are fixing a diet/recipe recommendation output that violated constraints.\n"
        "Rewrite it so it strictly satisfies the HARD CONSTRAINTS and keep the same sections: restaurants, breakfast, dinner, workouts, and 3 recipes.\n\n"
        "HARD CONSTRAINTS:\n"
        "- Veg vs Non-Veg is STRICT. If Veg, do NOT include any meat, fish/seafood, or eggs.\n"
        "- Do NOT include allergens listed in 'Person allergics'.\n"
        "- Do NOT include ingredients listed in 'Ingredients to avoid'.\n"
        "- Respect disease constraints.\n"
        "- If max cooking time is provided, keep each recipe within that time.\n\n"
        "Person age: {age}\n"
        "Person gender: {gender}\n"
        "Person weight: {weight}\n"
        "Person height: {height}\n"
        "Person veg_or_nonveg: {veg_or_nonveg}\n"
        "Person generic disease: {disease}\n"
        "Person region: {region}\n"
        "Person state or City: {state}\n"
        "Person allergics: {allergics}\n"
        "Person foodtype: {foodtype}\n"
        "Ingredients available: {ingredients_have}\n"
        "Ingredients to avoid: {ingredients_avoid}\n"
        "Max cooking time (minutes): {max_cook_time_minutes}\n\n"
        "Original (bad) output:\n{original_text}\n\n"
        "Now provide the corrected output ONLY (no preamble)."
    ),
)

if LLMChain is not None:
    chain_repair = LLMChain(llm=model, prompt=prompt_template_repair)
else:
    chain_repair = prompt_template_repair | model


NON_VEG_TERMS = {
    "chicken",
    "mutton",
    "beef",
    "pork",
    "fish",
    "salmon",
    "tuna",
    "prawn",
    "shrimp",
    "crab",
    "lobster",
    "seafood",
    "egg",
    "eggs",
    "turkey",
    "lamb",
    "bacon",
    "ham",
    "sausage",
    "pepperoni",
    "gelatin",
}


def _split_csv_terms(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw or raw.lower() == "not provided":
        return []
    # Split on commas/newlines, normalize whitespace
    parts = re.split(r"[\n,]+", raw)
    return [p.strip().lower() for p in parts if p.strip()]


def _contains_any_term(haystack_lower: str, terms: list[str] | set[str]) -> bool:
    for term in terms:
        if term and term in haystack_lower:
            return True
    return False

def load_css(file_name: str = "styles.css") -> None:
    css_path = Path(__file__).with_name(file_name)
    try:
        css = css_path.read_text(encoding="utf-8")
    except OSError:
        st.warning(f"CSS file not found: {css_path}")
        return

    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)


load_css()

# Create a Streamlit web app
st.title('PATIENT DIET RECOMMENDATION SYSTEM')

# User input form
age = st.text_input('Age')
gender = st.selectbox('Gender', ['Male', 'Female'])
weight = st.text_input('Weight (kg)')
height = st.text_input('Height (cm)')
veg_or_nonveg = st.selectbox('Veg or Non-Veg', ['Veg', 'Non-Veg'])
disease = st.text_input('Disease')
region = st.text_input('Region')
state = st.text_input('State / City')
allergics = st.text_input('Allergics')
foodtype = st.text_input('Food Type')

st.markdown("---")
st.subheader("Recipe recommendation (optional)")
ingredients_have = st.text_area(
    'Which ingredients do you have? (optional, comma-separated)',
    placeholder='e.g., oats, milk, banana, eggs, tomato',
)
ingredients_avoid = st.text_area(
    "Which ingredients do you NOT want in the recipe? (optional, comma-separated)",
    placeholder='e.g., peanuts, gluten, sugar',
)
max_cook_time_minutes = st.number_input(
    'Time to make (minutes) (optional)',
    min_value=0,
    step=5,
    value=0,
)

# Button to trigger recommendations
if st.button('Get Recommendations'):
    # Check if all form fields are filled
    if age and gender and weight and height and veg_or_nonveg and disease and region and state and allergics and foodtype:
        ingredients_have_value = (ingredients_have or "").strip() or "Not provided"
        ingredients_avoid_value = (ingredients_avoid or "").strip() or "Not provided"
        max_cook_time_value = (
            str(int(max_cook_time_minutes))
            if isinstance(max_cook_time_minutes, (int, float)) and max_cook_time_minutes > 0
            else "Not provided"
        )

        input_data = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'veg_or_nonveg': veg_or_nonveg,
            'disease': disease,
            'region': region,
            'state': state,  # Include state in input_data
            'allergics': allergics,
            'foodtype': foodtype,
            'ingredients_have': ingredients_have_value,
            'ingredients_avoid': ingredients_avoid_value,
            'max_cook_time_minutes': max_cook_time_value,
        }

        try:
            results = chain_resto.invoke(input_data)
        except Exception as e:
            msg = str(e)
            st.error(msg)
            if "NOT_FOUND" in msg and "models/" in msg:
                st.info(
                    "The configured Gemini model name may not be available for your API/project. "
                    "Set `GEMINI_MODEL` (or Streamlit Secret `GEMINI_MODEL`) to an available model, "
                    "e.g. `gemini-1.5-pro-latest` or `gemini-1.5-flash-latest`."
                )
            st.stop()

        # Extract recommendations
        if isinstance(results, dict) and "text" in results:
            results_text = results["text"]
        else:
            results_text = str(results)

        # Validate hard constraints and auto-repair if needed
        results_text_lower = results_text.lower()
        violations: list[str] = []

        if veg_or_nonveg.strip().lower() == "veg":
            if _contains_any_term(results_text_lower, NON_VEG_TERMS):
                violations.append("Non-veg items found while Veg was selected")

        allergics_terms = _split_csv_terms(allergics)
        if allergics_terms and _contains_any_term(results_text_lower, allergics_terms):
            violations.append("Allergen(s) appeared in output")

        avoid_terms = _split_csv_terms(ingredients_avoid_value)
        if avoid_terms and _contains_any_term(results_text_lower, avoid_terms):
            violations.append("Avoid-ingredient(s) appeared in output")

        if violations:
            st.warning(
                "Some recommendations didn’t match your constraints. "
                "Regenerating a corrected result…"
            )
            repair_input = dict(input_data)
            repair_input["original_text"] = results_text

            try:
                repaired = chain_repair.invoke(repair_input)
            except Exception:
                repaired = None

            if repaired is not None:
                if isinstance(repaired, dict) and "text" in repaired:
                    results_text = repaired["text"]
                else:
                    results_text = str(repaired)
        st.write("Generated Recommendations:")

        st.write(results_text)
    else:
        st.write("Sorry, you did not provide any information. Please fill in all the form fields.")


st.markdown(
    """
    <div class="footer">
        &copy; 2024 CODE_WIZARDS. All rights reserved.
        @Kartik Panganti
        @Zahid shaikh
        @Vijaykumar Maske
    </div>
    """,
    unsafe_allow_html=True,
)
