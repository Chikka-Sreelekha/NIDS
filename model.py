
# Libraries
import pandas as pd
import warnings
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score


warnings.filterwarnings('ignore')

print("<------- Dataset----------->")
print("****************************")
data = pd.read_csv("networkintrusion.csv")
print(data.head(10))
print()

print("<--------Data Preprocesing----------->")
print("***************************************")
data1=data.isnull().sum()
print(data1)
print()

  # Label Encoding
print("<-----Label Encoding------>")
print("***************************")

labelencoder=LabelEncoder()
data["Label"]=LabelEncoder().fit_transform(data["Label"])
data["Subflow_Bwd_Byts"]=LabelEncoder().fit_transform(data["Subflow_Bwd_Byts"])
print(data.head(10))
print()

print("<------ Data Splitting------>")
print("*****************************")

X = data[['Dst_Port','Protocol','Flow_Duration','Subflow_Bwd_Byts',
          'Idle_Std','Subflow_Fwd_Pkts','Bwd_Pkts','Flow_Byts']]
print(X.head(10))
y = data['Label']
print(y.head(10))
print()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

print("<------Normalize------>")
print("*****************************")
scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

print("<-------Algorithm implementation------>")
print("***************************************")
print()
print("Support Vector Machine")
svm = LinearSVC()
svm.fit(x_train, y_train)
predictions = svm.predict(x_test)

print("Accuracy Score : " , accuracy_score(predictions,y_test))
print()

print("<-------Confusion Matrix--------->")
print("**************************************")


print("Confusion Matrix")
print("****************")
 #calculate the confusion matrix
cf_matrix = confusion_matrix(predictions,y_test)
print(cf_matrix)
print()
print('Confusion matrix\n\n')
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix")
plt.show()          
print()


print("<-------Classification Report--------->")
print("**************************************")

print("Classification Report : \n\n" , classification_report(predictions,y_test),"\n")
print()

pickle.dump(svm, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)

print("<-----Prediction status------>")
print("******************************")


data=[[0.0117601,-0.0634753,-0.303096,0.763061,-0.110092,-0.103892,
       -0.0263787,-0.0677107]]


p=svm.predict(data)

if p==0:
    print("Malicious")
else:
    print("Benign")













