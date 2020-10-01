import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import KernelPCA , PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda 
from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        labelencoder=LabelEncoder()
        for col in data.columns:
            data[col] = labelencoder.fit_transform(data[col])
        return data
    
    @st.cache(persist=True)
    def split(df):
        y = df.iloc[:,0]
        x = df.iloc[:,1:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()

    # st.sidebar.subheader("Choose Dataset")
    # file_name = st.sidebar.selectbox("file_name",("mushroom.csv", "dataset_1.csv", "dataset_2.csv", "dataset_3.csv")


    # df = load_data(file_name)
    class_names = ['value_0', 'value_1']
    df = load_data()
    

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)


    X_train, X_test, y_train, y_test = split(df)

    st.sidebar.subheader("Dimenstion Reduction Technique")
    method = st.sidebar.selectbox("method", ("Principal Component Analysis (PCA)","Linear discriminant Analysis (LDA)","KernalPCA","NO REDUCTION"))

    if(method == "Principal Component Analysis (PCA)"):
        no_of_components = st.sidebar.number_input("no. of input feature", 1 , 5 , step= 1,key='n_components')
        red_tech = PCA(n_components=no_of_components)
        X_train = red_tech.fit_transform(X_train)
        X_test = red_tech.transform(X_test)


    if(method == "Linear discriminant Analysis (LDA)"):
        no_of_components = st.sidebar.number_input("no. of input feature", 1, 5, step = 1, key='n_components')
        red_tech = lda(n_components=no_of_components)
        X_train = red_tech.fit_transform(X_train,y_train)
        X_test = red_tech.transform(X_test)

    if(method == "KernalPCA"):
        no_of_components = st.sidebar.number_input("no. of input feature", 1, 5, step = 1, key='n_components')
        ker = st.sidebar.radio("kernel_selection", ("Linear","RBF"), key ='kern')
        red_tech = KernelPCA(n_components=no_of_components, kernel=ker)
        X_train = red_tech.fit_transform(X_train)
        X_test = red_tech.transform(X_test)


    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "Auto select acco. to Dataset"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        parameter = st.sidebar.radio("parameter_selection", ("Mannual","Auto"))
        if(parameter=="Mannual"):
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
            kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
            gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)

        if(parameter=="Auto"):
            model_a = SVC(kernel = 'rbf')
            model_a.fit(X_train, y_train)
            prameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':[1,10]}
            model = GridSearchCV( model_a, prameters, n_jobs = -1)
            model.fit(X_train, y_train)


        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        parameter = st.sidebar.radio("parameter_selection", ("Mannual","Auto"))
        if(parameter=="Mannual"):
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
            max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
            model = LogisticRegression(C=C,max_iter=max_iter)
            model.fit(X_train, y_train)

        if(parameter=="Auto"):
            model_a = LogisticRegression()
            model_a.fit(X_train, y_train)
            prameters = [{'C':[1,10],'max_iter':[100,500]}]
            model = GridSearchCV( model_a, prameters, n_jobs = -1)
            model.fit(X_train, y_train)


        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        parameter = st.sidebar.radio("parameter_selection", ("Mannual","Auto"))
        if(parameter=="Mannual"):
            n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
            max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
            bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(X_train, y_train)

        if(parameter=="Auto"):
            model_a = RandomForestClassifier(n_estimators = 100)
            model_a.fit(X_train, y_train)
            prameters = {'n_estimators':[100,300,10],'criterion':['gini','entropy'],'max_depth':[1,20],'bootstrap':['True','False']}
            model = GridSearchCV( model_a, prameters, n_jobs = -1)
            model.fit(X_train, y_train)

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            accuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    # if classifier == 'Auto select acco. to Dataset':
    #     metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    #     if st.sidebar.button("Classify", key='classify'):
    #         st.subheader("XG boost")
    #         model = XGBClassifier()
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy: ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics)

        
if __name__ == '__main__':
    main()