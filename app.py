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
st.image('https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg')
st.subheader('''User Must Enter Given Value to Predict Price:
        ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Feature') 
st.sidebar.image('https://www.providencejournal.com/gcdn/presto/2021/09/23/NPRJ/4ad329ab-09e1-41d0-a1d5-8b003da067c6-RIPRO-091020-NE_CONJURING_-Copy-.jpg?crop=5533,3113,x0,y0&width=3200&height=1801&format=pjpg&auto=webp')
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
     st.subheader(body)

    st.success(body)
else:
    body = 'Invalid House features Values'
    
    st.warning(body)


st.markdown('Designed by: **Nitin Kamia**')
    





