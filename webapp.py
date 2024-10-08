import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('welltrained_model.pkl', 'rb'))

data=pd.read_csv('diabetes.csv')


# creating a function for Prediction

def diabetes_prediction(input_data):
    
    
    input_df=pd.DataFrame([input_data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    input_df = input_df.apply(pd.to_numeric,errors='coerce')
    input_df=input_df.fillna(0)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_df.values.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    
    # giving a title
    st.title(	'💻Diabetes Prediction Web App')
    st.info('This is app predict the diabetes')
   
     
   


    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
