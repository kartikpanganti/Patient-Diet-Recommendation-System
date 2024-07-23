import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
import langchain.globals as lcg

# Set verbose to True or False based on your requirements
lcg.set_verbose(True)  # Enable verbose mode if needed

#  model template
os.environ["GOOGLE_API_KEY"] = 'AIzaSyC42poakq8cly-o_pK-g8gBgIZM9r-Yz1I'
generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = GoogleGenerativeAI(model="gemini-1.0-pro", generation_config=generation_config)
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
chain_resto = LLMChain(llm=model, prompt=prompt_template_resto)

# here is some my custom styling on streamlit
st.markdown(
    """
    <style>


    #MainMenu{
    visibility:hidden;
    }

    .stActionButton{
    visibility:hidden;
    }

    .st-emotion-cache-h4xjwg ezrtsby2{
    visibility:hidden;
    }
        .title {
            font-size: 25px;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
        }
        .content {
            font-family: 'Helvetica', sans-serif;
        #petient
        }
        span.st-emotion-cache-10trblm.e1nzilvr1{
            margin-top: 50px;
            color: darkmagenta;
                text-shadow: #FC0 1px 0 10px;

    text-align: center;
    font-size: 54px;
        }
       section.main.st-emotion-cache-bm2z3a.ea3mdgi8 {
    background-size: cover;
    background-image: url(https://img.freepik.com/free-vector/gradient-abstract-background_23-2149131346.jpg?w=1060&t=st=1721715245~exp=1721715845~hmac=63d8f83801288b55c60ea173b4b07b304a1ad4021767ea96baf2d95ce66dc055);
}

p .st-emotion-cache-1sno8jx e1nzilvr4{
{
    font-size: 50px;
    font-family: auto;
    color: #96f700;
    text-shadow: 5px 3px 4px darkmagenta;
}
}
h1 {
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 700;
    color: rgb(9 253 255);
    padding: 1.25rem 0px 1rem;
    margin: 0px;
    line-height: 1.2;
    font-size: 49px;
    text-align: center;
    text-shadow: 6px 6px 6px blue;
}

label.st-emotion-cache-1qg05tj.e1y5xkzn3 {
    color: #ffa1a1;
    font-weight: bold;
    text-shadow: 0px 0px 5px red;
}
button.st-emotion-cache-19rxjzo.ef3psqc12{
background-color: #008CBA;;
}


strong {
    color: #0fe800;
    font-size: 25px;
    text-shadow: #FC0 10px 1px 13px;
    font-weight: bold;
}

li {
    color: #000000;
    /* font-weight: bolder; */
    webkit-text-fill-color: white;
    -webkit-text-stroke: 1px #2fbcff;
}

li::marker{
list-style-type: circle;
color:#2fbcff

}

    </style>
    """,
    unsafe_allow_html=True,
)

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
state = st.text_input('State')
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

        results = chain_resto.invoke(input_data)

        # Extract recommendations
        results_text = results['text']
        st.write("Generated Recommendations:")

        st.write(results_text)
    else:
        st.write("Sorry, you did not provide any information. Please fill in all the form fields.")


st.markdown(
    """
    <div class="footer">
        &copy; 2024 Kartik Panganti. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)
