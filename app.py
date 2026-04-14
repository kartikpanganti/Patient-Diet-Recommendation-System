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
        gemini_model = st.secrets.get("GEMINI_MODEL")
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
    max_output_tokens=3072,
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
        'servings',
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
             "Preferred servings (number of persons) for recipes: {servings}\n\n"
             "Optional recipe recommendation inputs (may be empty):\n"
             "Ingredients available: {ingredients_have}\n"
             "Ingredients to avoid: {ingredients_avoid}\n"
             "Max cooking time (minutes): {max_cook_time_minutes}\n\n"
             "Also recommend 3 simple recipes that match the person’s diet preferences and disease constraints. "
             "If ingredients are provided, prefer using available ingredients; if exclusions are provided, do not use them. "
             "If max cooking time is provided, keep each recipe within that time.\n\n"
             "OUTPUT FORMAT (follow exactly; use Markdown):\n"
             "## Restaurants (6)\n"
             "- ...\n\n"
             "## Breakfast (6)\n"
             "- ...\n\n"
             "## Dinner (5)\n"
             "- ...\n\n"
             "## Workouts (6)\n"
             "- ...\n\n"
             "## Recipes (3)\n"
             "### Recipe 1: <Name>\n"
             "Servings: <number of persons>\n"
             "Time: Prep <X> min | Cook <Y> min | Total <Z> min\n"
             "Ingredients:\n"
             "- <quantity> <unit> <ingredient> (include household + metric where possible, e.g., 1 cup / 240 ml)\n"
             "Procedure:\n"
             "1. Start-to-end detailed steps (be explicit: prep, cook, timings, heat level, when to add what).\n"
             "2. ...\n"
             "Diet notes: 1–2 lines explaining why it fits the disease/goal and any safe swaps.\n\n"
             "Repeat the same structure for Recipe 2 and Recipe 3.\n"
             "Do NOT mention excluded ingredients at all (don’t even say 'avoid X')."
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
        'servings',
        'ingredients_have',
        'ingredients_avoid',
        'max_cook_time_minutes',
        'original_text',
    ],
    template=(
        "You are fixing a diet/recipe recommendation output that violated constraints.\n"
        "Rewrite it so it strictly satisfies the HARD CONSTRAINTS and keep the same sections and the same OUTPUT FORMAT.\n\n"
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
        "Preferred servings (number of persons) for recipes: {servings}\n"
        "Ingredients available: {ingredients_have}\n"
        "Ingredients to avoid: {ingredients_avoid}\n"
        "Max cooking time (minutes): {max_cook_time_minutes}\n\n"
        "Original (bad) output:\n{original_text}\n\n"
        "OUTPUT FORMAT (follow exactly; use Markdown):\n"
        "## Restaurants (6)\n"
        "- ...\n\n"
        "## Breakfast (6)\n"
        "- ...\n\n"
        "## Dinner (5)\n"
        "- ...\n\n"
        "## Workouts (6)\n"
        "- ...\n\n"
        "## Recipes (3)\n"
        "### Recipe 1: <Name>\n"
        "Servings: <number of persons>\n"
        "Time: Prep <X> min | Cook <Y> min | Total <Z> min\n"
        "Ingredients:\n"
        "- <quantity> <unit> <ingredient> (include household + metric where possible)\n"
        "Procedure:\n"
        "1. Start-to-end detailed steps\n"
        "Diet notes: 1–2 lines\n\n"
        "Repeat for Recipe 2 and Recipe 3.\n"
        "Do NOT mention excluded ingredients at all (don’t even say 'avoid X').\n\n"
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


def _normalize_unknown_text(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return "Not provided"
    lowered = raw.lower()
    if lowered in {"idk", "i dont know", "i don't know", "na", "n/a", "not sure", "unknown", "none", "no"}:
        return "Not provided"
    return raw

def _term_is_negated(text_lower: str, term: str, start_idx: int) -> bool:
    """Heuristic: ignore matches like 'no egg', 'egg-free', 'without eggs', 'free of egg'."""
    window_start = max(0, start_idx - 20)
    prefix = text_lower[window_start:start_idx]

    if re.search(r"\b(no|without|avoid|excluding)\s+$", prefix):
        return True

    if re.search(r"\bfree\s+of\s+$", prefix):
        return True

    # Handle hyphenated forms like 'egg-free'
    after = text_lower[start_idx:start_idx + len(term) + 6]
    if re.search(rf"\b{re.escape(term)}\s*[- ]\s*free\b", after):
        return True

    return False


def _contains_nonveg_violation(text: str, terms: set[str]) -> bool:
    text_lower = text.lower()
    for term in terms:
        for match in re.finditer(rf"\b{re.escape(term)}\b", text_lower):
            if not _term_is_negated(text_lower, term, match.start()):
                return True
    return False


def _extract_recipe_ingredients_blocks(text: str) -> list[str]:
    """Extract ingredient blocks from the 'Recipes' section (best-effort)."""
    blocks: list[str] = []
    # Match each recipe block starting at a recipe heading
    for recipe_match in re.finditer(r"^###\s*Recipe\s*\d+\s*:\s*.*$", text, flags=re.IGNORECASE | re.MULTILINE):
        start = recipe_match.start()
        next_match = re.search(
            r"^###\s*Recipe\s*\d+\s*:\s*.*$",
            text[recipe_match.end():],
            flags=re.IGNORECASE | re.MULTILINE,
        )
        end = len(text) if next_match is None else recipe_match.end() + next_match.start()
        recipe_text = text[start:end]

        ing_match = re.search(
            r"Ingredients\s*:\s*(.*?)\n\s*(Procedure|Steps)\s*:\s*",
            recipe_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if ing_match:
            blocks.append(ing_match.group(1))
    return blocks

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

st.caption("Tip: If you don’t know a value, you can type IDK / NA / Not sure.")

# User input form
age = st.text_input('Age', placeholder='e.g., 30 (or NA / IDK)')
gender = st.selectbox('Gender', ['Male', 'Female'])
weight = st.text_input('Weight (kg)', placeholder='e.g., 72 (or NA / IDK)')
height = st.text_input('Height (cm)', placeholder='e.g., 170 (or NA / IDK)')
veg_or_nonveg = st.selectbox('Veg or Non-Veg', ['Veg', 'Non-Veg'])
disease = st.text_input('Disease', placeholder='e.g., diabetes / BP / NA')
region = st.text_input('Region', placeholder='e.g., South India / NA')
state = st.text_input('State / City', placeholder='e.g., Pune / NA')
allergics = st.text_input('Allergics', placeholder='e.g., peanuts, milk (or NA if none/unknown)')
foodtype = st.text_input('Food Type', placeholder='e.g., low sugar / high protein / NA')

st.markdown("---")
st.subheader("Recipe recommendation (optional)")
servings = st.number_input(
    'Number of persons (servings) (optional)',
    min_value=1,
    step=1,
    value=2,
    help='If you are not sure, keep the default.',
)
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
    # Allow unknown inputs; normalize blanks/NA/IDK to "Not provided"
    age_value = _normalize_unknown_text(age)
    weight_value = _normalize_unknown_text(weight)
    height_value = _normalize_unknown_text(height)
    disease_value = _normalize_unknown_text(disease)
    region_value = _normalize_unknown_text(region)
    state_value = _normalize_unknown_text(state)
    allergics_value = _normalize_unknown_text(allergics)
    foodtype_value = _normalize_unknown_text(foodtype)

    ingredients_have_value = (ingredients_have or "").strip() or "Not provided"
    ingredients_avoid_value = (ingredients_avoid or "").strip() or "Not provided"
    max_cook_time_value = (
        str(int(max_cook_time_minutes))
        if isinstance(max_cook_time_minutes, (int, float)) and max_cook_time_minutes > 0
        else "Not provided"
    )

    input_data = {
        'age': age_value,
        'gender': gender,
        'weight': weight_value,
        'height': height_value,
        'veg_or_nonveg': veg_or_nonveg,
        'disease': disease_value,
        'region': region_value,
        'state': state_value,  # Include state in input_data
        'allergics': allergics_value,
        'foodtype': foodtype_value,
        'ingredients_have': ingredients_have_value,
        'ingredients_avoid': ingredients_avoid_value,
        'max_cook_time_minutes': max_cook_time_value,
        'servings': str(int(servings)) if isinstance(servings, (int, float)) else '2',
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
    violations: list[str] = []

    if veg_or_nonveg.strip().lower() == "veg":
        if _contains_nonveg_violation(results_text, NON_VEG_TERMS):
            violations.append("Non-veg items found while Veg was selected")

    # Check allergens/avoid only within the recipe ingredient lists to reduce false positives
    ingredient_blocks = _extract_recipe_ingredients_blocks(results_text)
    ingredient_text_lower = "\n".join(ingredient_blocks).lower()

    allergics_terms = _split_csv_terms(allergics_value)
    if allergics_terms and ingredient_blocks and _contains_any_term(ingredient_text_lower, allergics_terms):
        violations.append("Allergen(s) appeared in recipe ingredients")

    avoid_terms = _split_csv_terms(ingredients_avoid_value)
    if avoid_terms and ingredient_blocks and _contains_any_term(ingredient_text_lower, avoid_terms):
        violations.append("Avoid-ingredient(s) appeared in recipe ingredients")

    if violations:
        repair_input = dict(input_data)
        repair_input["original_text"] = results_text

        with st.spinner("Generating recommendations..."):
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
    st.markdown(results_text)


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
