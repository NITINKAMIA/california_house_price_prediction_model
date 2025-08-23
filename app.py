import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time
#title
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price Prediction')
st.header('Model of housing prices to predict median house values in California')
st.image('https://media.istockphoto.com/id/174631847/photo/beautiful-california-house-with-stone-walls.jpg?s=612x612&w=0&k=20&c=uS8hCmpYBMOV8P1dYWGRW1_XxEW6xKTzf-mDKmobWb4=')
st.subheader('''User Must Enter Given Value to Predict Price:
        ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Feature') 
st.sidebar.image('https://media.istockphoto.com/id/83802508/photo/stairs-leading-to-craftsman-house.jpg?s=612x612&w=0&k=20&c=Ai2VREsZR-l8XPf0Cn5VKputzmv0bSk4CoUUW3DZf1I=')
temp_df = pd.read_csv('california.csv')
random.seed(12)

all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]
st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicted Price')
place = st.empty()
place.image("https://media.lordicon.com/icons/wired/outline/19-magnifier-zoom-search.gif",width=80)

if price>0:
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
     #st.subheader(body)

    st.success(body)
else:
    body = 'Invalid House features Values'
    
    st.warning(body)


st.markdown('Designed by: **Nitin Kamia**')
    







