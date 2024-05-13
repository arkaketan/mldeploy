import streamlit as st
import pandas as pd
import numpy as np
import pickle
from Orange.data import *

app_mode = st.sidebar.selectbox('Select Page',['Home','Predict'])

def predictions(age,height_cm,weight_kg,eur_value,eur_wage,overall,pac,sho,pas,dri,defence,phy,international_reputation,weak_foot):
    # Load the model
    with open("football_predict4_new.pkcls", "rb") as f:
        model_loaded = pickle.load(f)
        
    # Define the domain with variable names
    domain = Domain([
        ContinuousVariable("age"),
        ContinuousVariable("height_cm"),
        ContinuousVariable("weight_kg"),
        ContinuousVariable("eur_value"),
        ContinuousVariable("eur_wage"),
        ContinuousVariable("overall"),
        ContinuousVariable("pac"),
        ContinuousVariable("sho"),
        ContinuousVariable("pas"),
        ContinuousVariable("dri"),
        ContinuousVariable("defence"),
        ContinuousVariable("phy"),
        ContinuousVariable("international_reputation"),
        ContinuousVariable("weak_foot")
    ])
    
    # Create a data table with the provided values
    data = Table(domain, [[age, height_cm, weight_kg, eur_value, eur_wage, overall, pac, sho, pas, dri, defence, phy, international_reputation, weak_foot]])
    
    # Make predictions
    if st.button("Predict"):
        output = model_loaded(data)
        preds = model_loaded.domain.class_var.str_val(output)
        preds = "Expected Rating of the Player: " + preds
        st.success(preds)

if app_mode == 'Home': 
    st.title('Player Potential Estimator')
    st.image('image.jpg')
    st.markdown('''The following are the steps to utilise this tool:--
    1. If accessing through PC/Laptop device - On your Left hand side of the page, under "Select Page", choose "Predict"
    via the drop down list. 
    2. If accessing through a mobile device, head to the top right corner of the screen, tap on the ">" button, from the 
    pane that shows up, follow the same step as the previous one.
    3. Use the sliders to input details for the player whose potential is to be determined.
    4. Click on "Predict" button at the end of the page to receive results. ''')
     
    

elif app_mode == 'Predict':
    st.title('Potential Prediction') 
    st.subheader('Fill in player details ')

    age = st.slider("Select Age of the player:", 15, 50, 15)
    height_cm = st.slider("Select height of the player:", 150, 210, 150)
    weight_kg = st.slider("Select weight of the player:", 40, 100, 50)
    eur_value = st.slider("Select current valuation of the player:", 10000000, 140000000, 10000000)
    eur_wage = st.slider("Select current weekly wage of the player:", 1000000, 100000000, 1000000)
    overall = st.slider ("Select overall player rating:", 40, 100, 40)
    pac = st.slider ("Select player's pace:", 40, 100, 40)
    sho = st.slider ("Select player's shooting accuracy:", 40, 100, 40)
    pas = st.slider ("Select player's passing accuracy:", 40, 100, 40)
    dri = st.slider ("Select player's dribbling rating:", 40, 100, 40)
    defence = st.slider ("Select player's defensive capability rating:", 40, 100, 40)
    phy = st.slider ("Select player physicality rating:", 40, 100, 40)
    international_reputation = st.slider ("Select international reputation rating:", 1, 5, 1)
    weak_foot = st.slider ("Select player's weak foot rating:", 1, 5, 1)
    
    predictions(age, height_cm, weight_kg, eur_value, eur_wage, overall, pac, sho, pas, dri, defence, phy, international_reputation, weak_foot)
