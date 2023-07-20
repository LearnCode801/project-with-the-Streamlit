from PIL import Image
import streamlit as st
from sklearn import metrics
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from pyforest import*
import warnings
warnings.filterwarnings("ignore")

# set title
st.title("Streamlit Web App-1 for Data View")
st.write("### SVM & KNN Algorithum with Differents Datasets")

# set logo
image = Image.open('logo 1.JPG')
st.image(image, use_column_width=True)

# set tag line
st.write("##### Hello to the future ")

# set side bar
data_set_name = st.sidebar.selectbox(
    "Select Dataset for model ", ("Iris", "Breast Cancer", "Wine"))
classifier_name = st.sidebar.selectbox("Select Model", ("KNN", "SVM"))

# Loading the data set


def get_data(name):
    data = None
    if name=='Iris':
        data=datasets.load_iris()
    elif name=='Wine':  
        data=datasets.load_wine() 
    else:    
        data=datasets.load_breast_cancer() 
    x=data.data
    y=data.target
    fn=data.feature_names
    tn=data.target_names
    # st.write(data.describe())
    return x,y,fn,tn
	
# load and show the dataset
x, y ,fn,tn= get_data(data_set_name)
fn=list(fn)
tn=list(tn)
x=pd.DataFrame(x,columns=[i for i in fn])
y=pd.DataFrame(y,columns={'target'})

st.write("##### The indenpendent columns of in the dataset =>> {}".format(data_set_name))
st.dataframe(x)

st.write("##### The denpendent column of in the dataset ==>  {}".format(data_set_name))
st.dataframe(y)

st.write("The independent variable are :")
st.write(fn)

st.write("The dependent independent are :")
st.write(tn)

st.write("The shape of data:")
shp=(x.shape + y.shape)
st.write(shp)

# st.write("###### box plot of the independent data set")
# fig=plt.figure()
# sns.boxplot(data=x,orient='h')
# st.pyplot(fig)

# st.write("###### Hist plot of the independent data set")
# fig=plt.figure()
# plt.hist(x)
# st.pyplot(fig)


# st.write("###### Dist plot of the independent data set")
# fig=plt.figure()
# sns.distplot(x)
# st.pyplot(fig)

# st.write("###### sacter plot of the independent data set")
# fig=plt.figure()
# sns.scatterplot(x=x.iloc[2], y=x.iloc[1],hue=x.iloc[1],data=x)
# st.pyplot(fig)

# st.write("###### Pairplot of the independent data set")
# fig=plt.figure()
# sns.pairplot(data=x)
# st.pyplot(fig)


# st.write("###### Heatmap of the independent data set")
# fig=plt.figure(figsize=(35,15))
# sns.heatmap(x.corr(),vmax=1, square=True,annot=True, cmap='viridis')
# plt.title('Correlation between different attributes')
# plt.show()
# st.pyplot(fig)

st.write("###### Correlation  of the independent data set ")
st.write(x.corr())

st.write("###### describe of the independent data set ")
st.write(x.describe().T)

# Building the Algorithum
# Selecting the hyper parameter

def hyper_parameter(name_of_clf):
    params={}
    if name_of_clf=='KNN':
        k=st.sidebar.slider("Choose k ",1,20,4)
        params['K']=k
    else:
        c=st.sidebar.slider("Choose c ",1,50,6)
        params['C']=c

    # here in this function we are returning the params dictionary
    return params
#here we are receing the dictionary in the parameter
parameter=hyper_parameter(classifier_name)

# selecting the model and then the model object
def get_model(name_of_clf):
    model=None
    if name_of_clf=='SVM':
        model= svm.SVC(C=parameter['C'])
    else:
        model=KNeighborsClassifier(n_neighbors=parameter['K'])
    return model   

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

model=get_model(classifier_name)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
st.write(y_pred)

st.write("###### The selected model   ->   {}".format(classifier_name))
st.write("###### The selected model   ->   {}".format(data_set_name))

score1=model.score(x_train,y_train)
st.write('###### the score at train data:   {}%'.format(score1*100))

score2=model.score(x_test,y_test)
st.write('###### the score at test data:   {}%'.format(score2*100))

# def confussion_metrics(name_of_clf):
#     if name_of_clf=='SVM':
#         cm=metrics.confusion_matrix(y_test,y_pred,labels=lab)
#         df_cm=pd.DataFrame(cm, index=[i for i in lab], columns=[i for i in plab])
#         plt.figure(figsize=(20,20))
#         sns.heatmap(df_cm,annot=True,fmt='g',cmap='PiYG')
#     else:
#         cm=metrics.confusion_matrix(y_test,y_pred,labels=['M','B'])
#         df_cm=pd.DataFrame(cm, index=[i for i in ["M","B"]], columns=[i for i in ["Predict M",'predict B']])
#         plt.figure(figsize=(7,5))
#         sb.heatmap(df_cm,annot=True)
#         sb.heatmap(cm,annot=True)    


    
    