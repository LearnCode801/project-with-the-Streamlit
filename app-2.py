from PIL import Image
import streamlit as st
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from pyforest import*
import warnings
warnings.filterwarnings("ignore")
 
# set title
st.title("Streamlit Web App-2")
st.write("### Uploading the CSV File and Perform the EDA,Viuilization,Modeling")
# set logo
image = Image.open('logo 1.JPG')
st.image(image, use_column_width=True)

# set tag line
st.write("##### Hello to the future ")

# set side bar
option1 = st.sidebar.selectbox(
    "Choose option", ("About Us", "EDA", "Viuilization", "Model"))


# Uploading the csv file
upload_file = st.file_uploader("Choose the csv file", type='csv')
flag=False
if upload_file is not None:
    st.success("successfully uploaded")
    flag=True
else:
    st.info("the file you uploaded is emty please a upload valid file")    



if (flag==True) & (option1=='EDA'):
    st.write("#### Explotry Data Analysis")
    data = pd.read_csv(upload_file)
    st.write("##### Head of your Uploaded Dataset")
    st.write(data.head(10))
    
    if st.checkbox("Shape of Dataset"):
        st.write(data.shape)
    if st.checkbox("Columns of Dataset"):
        st.write(data.columns)
    if st.checkbox("Null count values in Dataset's columns"):
        st.write(data.isna().sum())
    if st.checkbox("Dtypes of Dataset's columns"):
        st.write(data.dtypes)
    if st.checkbox("Correlation of Dataset columns"):
        st.write(data.corr())
    if st.checkbox("Describe of Dataset columns"):
        st.write(data.describe().T)
        
elif(flag==True) & (option1=='Viuilization'):
    st.write("#### Dataset Viuilization")
    data = pd.read_csv(upload_file)
    st.write("##### Head of your Uploaded Dataset")
    st.write(data.head(10))
    
    col=data.describe().columns
    # st.write(col)
    selected_columns=st.multiselect("Selecting the multiple column: ",(i for i  in col))
    st.write("selcected columns : ",selected_columns)
    
    if st.checkbox("Box Plot"):
        # data=col
        fig=plt.figure()
        sns.boxplot(data=data[selected_columns],orient='h')
        st.pyplot(fig)
        
    if st.checkbox("Dist PLot"):
        fig=plt.figure()
        sns.distplot(data[selected_columns])         
        st.pyplot(fig)
        
    if st.checkbox("Piegraph"):
        fig=plt.figure()
        pieplot=data[selected_columns].value_counts().plot.pie(autopct="%1.1f%%")
        st.write(pieplot)
        st.pyplot(fig)
        
    if st.checkbox("Heatmap"):
        fig=plt.figure(figsize=(35,15))
        sns.heatmap(data[selected_columns].corr(),vmax=1, square=True,annot=True, cmap='viridis')
        plt.title('Correlation between different attributes')
        plt.show()
        st.pyplot(fig)

        
    if st.checkbox("Pairplot"):
         fig=plt.figure()
         sns.pairplot(data=data[selected_columns])
         st.pyplot(fig)
        
    if st.checkbox("Histplot"):
        fig=plt.figure()
        plt.hist(data[selected_columns])
        st.pyplot(fig)
    
elif(flag==True) & (option1=='Model'):
    st.write("#### Appling the Model")
    data = pd.read_csv(upload_file)
    st.write("##### Head of your Uploaded Dataset")
    st.write(data.head(10))
    
    select_model = st.sidebar.selectbox("Select Model ", ("KNN","SVM" ,"LR", "Naive bayes","Decision Tree"))
    select_seed=st.sidebar.slider("Choose seed ",1,20,4)    
    
    def hyper_parameter(name_of_model):
        params={}
        if name_of_model=='KNN':
            k=st.sidebar.slider("Choose k ",1,20,4)
            params['K']=k
        elif name_of_model=='SVM':
            c=st.sidebar.slider("Choose c ",1,50,6)
            params['C']=c
        elif name_of_model=='Decision Tree':
            d=st.sidebar.slider("Choose depth of tree ",1,20,3)
            params['D']=d    
        else:
            st.sidebar.write("No more hyper parameter for other alogoritm")
        # here in this function we are returning the params dictionary
        return params
    #here we are receing the dictionary in the parameter
    parameter=hyper_parameter(select_model)
    
    # selecting the model and then the model object
    def get_model(name_of_model):
        model=None
        if name_of_model=='KNN':
            model=KNeighborsClassifier(n_neighbors=parameter['K'])
        
        elif name_of_model=='SVM':
            model= svm.SVC(C=parameter['C'])
        
        elif name_of_model=='LR':
            model=LogisticRegression(solver='liblinear')
            
        elif name_of_model=='Naive bayes':
            model=GaussianNB()
            
        elif name_of_model=='Decision Tree':
            model=DecisionTreeClassifier(criterion='gini' ,random_state=select_seed)
                    
        return model 
    


    x =data.drop(data.columns[-1],axis=1)
    y= data[data.columns[-1]]

    st.write(x)
    st.write(y)

    from scipy.stats import zscore
    Xscaled=x.apply(zscore)
    x=pd.DataFrame(Xscaled)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state= select_seed )

    model=get_model(select_model)

    model.fit(x_train,y_train)
    
    y_pred=model.predict(x_test)
    y_pred=pd.DataFrame(y_pred,columns=['y_pred'])
    st.write(y_pred)
    st.write(y_test)

    st.write("###### The selected model   ->   {}".format(select_model))

    score1=model.score(x_train,y_train)
    st.write('###### the score at train data:   {}%'.format(score1*100))

    score2=model.score(x_test,y_test)
    st.write('###### the score at test data:   {}%'.format(score2*100))
    
    lab=['0','1']
    cm=metrics.confusion_matrix(y_test,y_pred,labels=lab)
    df_cm=pd.DataFrame(cm, index=[i for i in lab], columns=[i for i in ['0','1']])
    fig=plt.figure(figsize=(10,10))
    sns.heatmap(df_cm,fmt='g', square=True,annot=True, cmap='viridis')
    st.pyplot(fig)

else:
    st.balloons()









