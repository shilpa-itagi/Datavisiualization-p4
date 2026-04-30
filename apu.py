import streamlit as st
import numpy as np

model =  pickle.load(open('iris_model.pkl','rb'))

st.title('Iris prediction and visualization')

sepal_length = st.slider("sepal Length",4.0,8.0,5.0)
sepal_width = st.slider("sepal width",2.0,4.5,3.0)

petal_length = st.slider("petal Length",1.0,7.0,4.0)
petal_width = st.slider("petal width",0.1,2.5,1.0)

input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
prediction = model.predict(input_data)

species = ['setosa','versicolor','verginica']

st.header('prediction')
st.success(species[prediction[0]])
