
import streamlit as st
import pickle
import numpy as np
import base64
lr = pickle.load(open("Linear_Regressor1.pkl", "rb"))
dt = pickle.load(open("Decision_Tree1.pkl", "rb"))
rf = pickle.load(open("Random_Forest1.pkl", "rb"))
##@st.experimental_memo
#def get_image_as_base64(file):
 #   with open 
page_bg_image ='''

<style>
 [data-testid="stAppViewContainer"]{
  background-image: url('Orangebg.jpg');
  background-size :cover;
}
</style>
'''
st.markdown(page_bg_image,unsafe_allow_html=True)

st.title('Insurance Charge Prediction Web app')
st.subheader('Fill the detail to predict insurance charges')

from PIL import Image
img = Image.open('198168-insurrace.jpg')
st.image(img)

model = st.sidebar.selectbox('Choose the ML model', ['Lin_reg', 'DT_reg', 'RF_reg'])

age = st.slider('Age', 18, 64, 1)
sex = st.selectbox('Sex', ['Male', 'Female'])
bmi = st.slider('BMI', 15, 53, 1)
children = st.selectbox('Children', [0, 1, 2, 3, 4, 5])
smoker = st.selectbox('Smoker', ['Yes', 'No'])
region = st.selectbox('Region', ['NorthWest', 'NorthEast', 'SouthWest', 'SouthEast'])

nwest = 0
neast = 0
swest = 0
seast = 0

if st.button('Predict Insurance Charges'):
    sex_num = 1 if sex == 'Male' else 0
    smoker_num = 1 if smoker == 'Yes' else 0

    if region == "NorthWest":
        nwest = 1
        neast = 0
        swest = 0
        seast = 0
    elif region == "SouthWest":
        nwest = 0
        neast = 0
        swest = 1
        seast = 0
    elif region == "SouthEast":
        nwest = 0
        neast = 0
        swest = 0
        seast = 1
    else:
        nwest = 0
        neast = 1
        swest = 0
        seast = 0

    test = np.array([age, sex_num, bmi, children, smoker_num, nwest, swest, seast])
    test = test.reshape(1, -1)

    if model == "Lin_reg":
        st.success(lr.predict(test)[0])
    elif model == "DT_reg":
        st.success(dt.predict(test)[0])
    else:
        st.success(rf.predict(test)[0])


st.write("Thanks for visiting Website")