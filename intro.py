
import streamlit as st

st.title("Our First App Based on the Sreamlet")
st.subheader("Predict about something you want ")

from PIL import Image
image=Image.open('logo.JPG')
st.image(image,use_column_width=True)

image=Image.open('logo 2.PNG')
st.image(image,use_column_width=True)


st.write("writing the text")

st.markdown("This is the Markdown ")
st.success("You successfully login ")
st.warning("something wrong happence")
st.help(range)
st.info("this is my website ")

import numpy as np
import pandas as pd

data=np.random.rand(5,20)
st.dataframe(data)


df=pd.DataFrame(np.random.rand(5,20),columns = ('col %d' % i for i in range(20)))
st.dataframe(df.style.highlight_max(axis=1))

st.text("----"*21)

chart_data=pd.DataFrame(np.random.randn(15,5),columns=['a','b','c','d','e'])
st.line_chart(chart_data)

st.text("////"*21)
st.area_chart(chart_data)

chart_data=pd.DataFrame(np.random.randn(3,5),columns=['a','b','c','d','e'])
st.bar_chart(chart_data)

# pip install matplotlib
import matplotlib.pyplot as plt
arr=np.random.normal(1,1,size=100)
plt.hist(arr,bins=20)
st.pyplot()


# import plotly
# import plotly.figure_factory as ff

# # Adding distplot
# x1=np.random.randn(200)%2
# x2=np.random.randn(200)
# x3=np.random.randn(200)%2

# hist_data=[x1,x2,x3]
# group_labels=['Group1','Group2','Group3']
# fig=ff.create_distplot(hist_data,group_labels,bin_size=[.2,.25,.5])

# st.plotly_chart(fig,use_container_width=True)

st.text("////"*20)

df=pd.DataFrame(np.random.randn(100,2)/[50,50]+[37.76,-122.4],columns=['lat','lon'])
st.map(df)

if st.button("Say hello"):
	 st.write("hello is here")
else: 
     st.write("why are you here")


genre=st.radio("What is your favourite genre?",('Commady','Drama','Film','Doumentry'))

if genre=='Commady':
	st.write("Good")
# elif genre=='Drama':
#     st.write('Cool')
else:
    st.write("No")    	

# Select box
optionbox=st.selectbox("SElecting the singgle options ",('Fantastic','Good','Batter'))
st.write("your said your night : ",optionbox)


optionbox=st.multiselect("SElecting the multiple option:: ",('Fantastic','Good','Batter'))
st.write("your said your night : ",optionbox)

age=st.slider('How Old Are Your??',0,100,30)
st.write('Your age is ',age)

value=st.slider("Select range :",0,200,(15,100))
st.write("the range is ", value)

number=st.number_input("Enetr the any number you want")
st.write('the number that youenter is :',number)

upload_file=st.file_uploader("Choose the csv file",type='csv')
if upload_file is not None:
	data=pd.read_csv(upload_file)
	st.write(data)
	st.success("successfully uploaded")
else:
    st.info("the file you uploaded is emty please a upload valid file")	


# color picker
color=st.color_picker("Pick your favourite color",'#fff')
st.write("Your favourite color is :",color)

st.text("\\\\\\"*10)
# Sidebar
add_sidebar=st.sidebar.selectbox("Select your model:",('select ...','KNN','XGB','RandomForest'))



import time 
my_bar=st.progress(0)
for percent_complete in range(100):
	time.sleep(0.1)
	my_bar.progress(percent_complete+1)

st.text("\\\\\\"*10)

with st.spinner('wait for it ...'):
    time.sleep(5)
st.success('successfully')

st.text("\\\\\\"*10)

st.ballooes()
