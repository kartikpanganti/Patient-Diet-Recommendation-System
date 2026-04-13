import os
from pathlib import Path
import streamlit as st
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
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'state', 'allergics', 'foodtype'],
    template="Diet Recommendation System:\n"
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
             "Person foodtype: {foodtype}."
)

if LLMChain is not None:
    chain_resto = LLMChain(llm=model, prompt=prompt_template_resto)
else:
    chain_resto = prompt_template_resto | model

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

# Button to trigger recommendations
if st.button('Get Recommendations'):
    # Check if all form fields are filled
    if age and gender and weight and height and veg_or_nonveg and disease and region and state and allergics and foodtype:
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
            'foodtype': foodtype
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
