import pickle
import numpy as np
import streamlit as st


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",

        # "United States of America",                               
        # "India",                                                   
        # Germany                                                  
        # United Kingdom of Great Britain and Northern Ireland     
        # Canada                                                   
        # France                                                   
        # Brazil                                                   
        # Poland                                                   
        # Netherlands                                              
        # Spain                                                    
        # Italy                                                    
        # Australia                                                
        # Russian Federation                                       
        # Turkey                                                   
        # Sweden                                                   
        # Switzerland                                               
        # Austria                                                   
        # Israel                                                    
        # Iran, Islamic Republic of...                              
        # Pakistan                                                  
        # Czech Republic                                            
        # China                                                     
        # Belgium                                                   
        # Bangladesh                                                
        # Ukraine                                                   
        # Romania                                                   
        # Mexico                                                    
        # Portugal                                                  
        # Greece                                                    
        # Denmark                                                   
        # Indonesia                                                 
        # Argentina                                                 
        # Nigeria                                                   
        # South Africa                                              
        # Norway                                                    
        # Finland                                                   
        # Hungary                                                   
        # New Zealand                                               
        # Egypt 
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    if ok := st.button("Calculate Salary"):
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")