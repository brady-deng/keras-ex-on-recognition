import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets,cross_validation,metrics
from sklearn import preprocessing
from tensorflow.contrib import learn
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv("mpg.csv",header=0)
df['displacement'] = df['displacement'].astype(float)
X = df[df.columns[1:8]]
y = df['mpg']

plt.figure()
f,ax1 = plt.subplots()
for i in range(1,8):
    number = 420+i
    ax1.locator_params(nbins=3)
    ax1 = plt.subplot(number)
    plt.title(list(df)[i])
    ax1.scatter(df[df.columns[i]],y)
plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1.0)
plt.show()
x_train,x_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25)
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
model = Sequential()
model.add(Dense(10,input_dim=7,kernel_initializer='normal',activation = 'relu'))
model.add(Dense(5,kernel_initializer='normal',activation='relu'))
model.add(Dense(1,kernel_initializer='normal'))

model.compile(loss = 'mean_squared_error',optimizer='adam')
model.fit(x_train,y_train,nb_epoch=1000,validation_split=0.33,shuffle=True,verbose=2)
score = metrics.mean_squared_error(model.predict(scaler.transform(x_test)),y_test)
print("Total Mean Squared Error:"+str(score))
