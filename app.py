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
place.image("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQMAAADCCAMAAAB6zFdcAAAAolBMVEX6+/z////V19jZ29z19vfc3t/j5OXq6+zm5+jv8fI5upHS1NXBwMDz9PX5+vuqqqq5ubn0+vfR6+Dh8uuo28e64tI5wZZRT1CSkZJLSUrLy8xtbG2Ui48ge142souxsbIzqoVhYGGfn6BZWFlIRUaJiIlxcHCWlpd+fn92dXbG5tnZ7uafl5sfhmdjXF4/RUM2NzePhooXaE8rknETSDgyQDuxSgsEAAAEH0lEQVR4nO3aaXebOBiGYQQCpIBk2k684GYodh0n02W2zv//a4MEEmJpcpr2mLp6rg85zRs3htsgYztBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADBCrKW3ZCmElNt6U+/rzf4Qe5mB8O1O7bqWbXerwLcKhGyPIXHlu8qvCCTZlWTsUFOPKpByIycJCInfp95EIGXd7HHhqriKILyJQOKjOv8T2QuqVdpGWHrjLmVDVQPhngYFaSNEey8OBLLN9fMAHzYglY6wLa8oQmpwO5IzM2Fn1PzHPZk2OORFftL/2pgGkhr9ndqRnI6cm12IDKNOv8PczEJuZ8zcLOLtRpJtONOg2XGy0v9Y5V2EzIjtL6N2xoTZDjtK+vAXIu2+pYG5c25nVJpt7BsQ0UaoyVwDtff6K63bBiQLO0yaOxBmlnHR3YG0o5ReOkLfII7j7jDk7qy7HXNmid61w9MNyH27K32DODUPsW0QJmnSng7SHV34dOgbqId93KAxaRBFusE5sg1uf9Nuhw1W7arYN1AP+7hBMwtGDZrRpRuEcaIMG+hZPGyQ2JlucE+fOQ7C87BBNNMgmjaIFmigD/g4HjWwo75BP9MN9sQ2+PDa8cY04A9ugyzl0waCjRtkPF3kOGCdJLQNEjOLnePAzHQDsySqBm9cr0wDuncbxHRyLjRZsnEDpn60wHHwovVAOg3mzwU5aCCjSYNITtcDsuR6oITD9aA9/20De7PJcdCtia3X9jh4tA2SiKmHfNCAhSIZNUhCzhZuEH9LA3c9mD8O2NY2YCSIRs8LmWiulIYN1E+zRRp0a6Je8IZrYrsG2gb9TDd4SG2DP24dH0yDvLINmpUumzw30mT83JgJtsxz40uvD4rcNnjlMtfK5KRPma9fH2TT6wMzupIGQXr/zLlwHF8r/8TXSP2+hd/SgOzFkw2y8/U0iB3dK1kRObPudqkzS/W+lecnG9TmlU/m6GbUnXXb4Y4u//L5hUgtnmiQncz7B5RO3xqYmc3d7KdHIr0iDN9Lsw028vnf8AsgB7W3ZZGNqHPk8eGK3kr7HuRRPT8mY5KQj5/u3voSYV+MTwTt7d167U+E02lagH6+W9/ceBQhn3zgWB3LP3+/URH+9iWCPG+K/o8w+GFTNV//8itCQIKqPp6qvMxXD5vHUr+97l0EdRQkZVEVpXop2U6EdxGC7k+SnG+5Z2vCHEQIdIR3KsInX64Y59gIfh8JO0QwEf7JfI4gdIR/PftrvSEi3r9br7+wpbdjUc0LqC//5T4fBsrw2gkAAOBXJylPU8b6N9YZS1MurueToh+AJ4n+CNI2aL9jfnzKZElKBbeEoNKzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANb/pulGfLyZBHYAAAAASUVORK5CYII=",width=80)

if price>0:
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    # st.subheader(body)

    st.success(body)
else:
    body = 'Invalid House features Values'
    
    st.warning(body)

    