import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder


st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:turquoise;padding:18px">
<h1 style="color:brown;text-align:center;">Car Price Prediction App </h1>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)


age=st.sidebar.selectbox("What is the age of your car?",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car?", 0,350000, step=1000)
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
gears = st.sidebar.selectbox("Select gears:", (5,6,7,8))
car_model=st.sidebar.selectbox("Select model of your car:", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))
displacement = st.sidebar.slider("Select displacement:", 850, 3000, step=10)
fuel = st.sidebar.selectbox("Select fuel type:", ('Diesel', 'Benzine', 'Electric', 'LPG/CNG'))
drive_chain = st.sidebar.radio("Select drive chain:", ('front', '4WD', 'rear'))

model=pickle.load(open("final_model","rb"))
transformer = pickle.load(open('transformer_final', 'rb'))


my_dict = {
    "age": age,
    "hp_kW": hp,
    "km": km,
    'Gearing_Type':gearing_type,
    "make_model": car_model,
    "Gears" : gears,
    "Displacement_cc" : displacement,
    "Fuel" : fuel,
    "Drive_chain" : drive_chain }

df = pd.DataFrame.from_dict([my_dict])


st.header("The features of your car are follows:")
st.dataframe(df)

df2 = transformer.transform(df)

st.markdown("Press **PREDICT** to make prediction")

if st.button("PREDICT"):
    prediction = model.predict(df2)
    st.success("The estimated PRICE is ${}. ".format(int(prediction[0])))


