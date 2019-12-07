import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from scipy.stats import levy
from sklearn.preprocessing import MinMaxScaler
from algorithms import simulated_annealing,GeomDecay,random_hill_climb
from neural import NetworkWeights,ContinuousOpt
from activation import relu
from sklearn.metrics import accuracy_score
from datetime import datetime
import tkinter as tk 

def get_weights(weights):
    #sws = [np.reshape(weights[0:96],(24,4)),np.reshape(weights[96:100],4),np.reshape(weights[100:116],(4,4)),np.reshape(weights[116:120],4),np.reshape(weights[120:124],(4,1)),weights[124]]
    sws = [np.reshape(weights[0:96],(24,4)),np.reshape(weights[96:100],4),np.reshape(weights[100:116],(4,4)),[0.,0.,0.,0.],np.reshape(weights[116:120],(4,1)),[0.]]
    return sws

def predict(weights,X_test,Y_test,model):
    sws = get_weights(weights)
    model.set_weights(sws)
    y_pred = model.predict(X_test)
    # setting a confidence threshhold of 0.9
    y_pred_labels = list(y_pred > 0.9)

    for i in range(len(y_pred_labels)):
        if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
        else : y_pred_labels[i] = 0
    return accuracy_score(Y_test,y_pred_labels)

df = pd.read_csv('Eyes.csv')
X = df[df.columns[1:25]]
Y = df[df.columns[25]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.10,random_state=42)

sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

# Create model
model = Sequential()
model.add(Dense(4,input_dim=X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

def EagleStrategyWithSA(model,iters,rate):
    mn = MinMaxScaler(feature_range=(-3,3))
    r = levy.rvs(size=iters)
    max_accuracy = 0
    final_weights = []
    count = 1
    for i in list(r):
        weights = np.random.uniform(-1,1,120)
        weights = weights * i
        weights = mn.fit_transform(weights.reshape(-1,1))
        acc = predict(weights,X_test,Y_test,model)
        if acc > 0.5:
            clip_max = 5
            fitness = NetworkWeights(X_train,Y_train,[25,4,4,1],relu,True,True,rate)
            problem = ContinuousOpt(120,fitness,maximize=False,min_val=-1*clip_max,max_val=clip_max,step=rate)
            #print(problem.get_length(),len(weights))
            fitted_weights, loss = simulated_annealing(problem,schedule=GeomDecay(),max_attempts=50,max_iters=300,init_state=weights,curve=False)
            acc = predict(fitted_weights,X_test,Y_test,model)
            if acc > max_accuracy:
                max_accuracy = acc
                final_weights = fitted_weights
            if max_accuracy > 0.95:
                break
            count+=1
    return max_accuracy,count,final_weights        
'''start = datetime.now()
accuracy,count,final_weights = EagleStrategyWithSA(model,20,0.2)
print("Time taken ",datetime.now()-start)
print(accuracy,count)'''
def EagleStrategyWithHillClimbing(model,iters,rate):
    mn = MinMaxScaler(feature_range=(-3,3))
    r = levy.rvs(size=iters)
    max_accuracy = 0
    final_weights = []
    count = 1
    for i in list(r):
        weights = np.random.uniform(-1,1,120)
        weights = weights * i
        weights = mn.fit_transform(weights.reshape(-1,1))
        acc = predict(weights,X_test,Y_test,model)
        if acc > 0.5:
            clip_max = 5
            fitness = NetworkWeights(X_train,Y_train,[25,4,4,1],relu,True,True,rate)
            problem = ContinuousOpt(120,fitness,maximize=False,min_val=-1*clip_max,max_val=clip_max,step=rate)
            #print(problem.get_length(),len(weights))
            fitted_weights, loss = random_hill_climb(problem, max_attempts=10, max_iters=1000, restarts=0,
                      init_state=weights, curve=False, random_state=None)
            acc = predict(fitted_weights,X_test,Y_test,model)
            if acc > max_accuracy:
                max_accuracy = acc
                final_weights = fitted_weights
            if max_accuracy > 0.95:
                break
            count+=1
    return max_accuracy,count,final_weights        

#algorithm=""
def simulated():
    start = datetime.now()
    accuracy,count,final_weights = EagleStrategyWithSA(model,40,0.2)
    end=datetime.now()-start
    #print("Time taken ",datetime.now()-start)
    #print(accuracy,count)
    master = tk.Tk()
    master.title("Simulated Annealing Results")
    #x = sensor_value #assigned to variable x like you showed
    master.minsize(width=400,height=50)
    w = tk.Label(master, text="Time taken : "+str(end)+"\n"+"Accuracy : "+str(accuracy)) #shows as text in the window
    #w = tk.Label(master, text="Accuracy : "+str(accuracy))
    w.pack() #organizes widgets in blocks before placing them in the parent.          
    master.mainloop()
def rhc():
    start = datetime.now()
    accuracy,count,final_weights = EagleStrategyWithHillClimbing(model,50,0.2)
    end=datetime.now()-start
    master = tk.Tk()
    master.title("Random Hill Climbing Results")
    #x = sensor_value #assigned to variable x like you showed
    master.minsize(width=400, height=50)
    w = tk.Label(master, text="Time taken : "+str(end)+"\n"+"Accuracy : "+str(accuracy)) #shows as text in the window
    #w = tk.Label(master, text="Accuracy : "+str(accuracy))
    w.pack() #organizes widgets in blocks before placing them in the parent.          
    master.mainloop()
def backprop():
    startTime=datetime.now()
    # Data Normalization
    
    opt = 'adam'
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


    model.fit(X_train, Y_train, batch_size = 2, epochs = 50, validation_data = (X_test, Y_test), verbose = 2)
    end=datetime.now()-startTime

    y_pred = model.predict(X_test)

    y_pred_labels = list(y_pred > 0.9)

    for i in range(len(y_pred_labels)):
        if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
        else : y_pred_labels[i] = 0


    from sklearn.metrics import confusion_matrix,accuracy_score
    cm = confusion_matrix(Y_test, y_pred_labels)
    print("\n")
    print("Confusion Matrix : ")
    print(cm)
    print("\n")
    accuracy = accuracy_score(Y_test, y_pred_labels)

    df_results = pd.DataFrame()
    df_results['Actual label'] = Y_test
    df_results['Predicted value'] = y_pred
    df_results['Predicted label'] = y_pred_labels
    df_results.to_csv('Results.csv')

    master = tk.Tk()
    master.title("Backpropagation Results")
    master.minsize(width=400,height=50)
    w = tk.Label(master, text="Time taken : "+str(end)+"\n"+"Accuracy : "+str(accuracy)) #shows as text in the window
    w.pack() #organizes widgets in blocks before placing them in the parent.          
    master.mainloop()
r = tk.Tk() 
r.configure(background='black')
r.attributes("-fullscreen", True) 
r.title('AI Project') 
w = tk.Label(r, text="-"*1000)
w1 = tk.Label(r, text="ARTIFICIAL INTELLIGENCE PROJECT")
w1_2 = tk.Label(r, text="COMPARISON OF OPTIMIZATION ALGORITHMS")
w2 = tk.Label(r, text="-"*1000)
w3 = tk.Label(r, text="Gaurav C G, PES1201700989")
w4 = tk.Label(r, text="Sanjay Chari, PES1201700278")
w5 = tk.Label(r, text="Cyrus D'Lima, PES1201700228")
w6 = tk.Label(r, text="-"*1000)
button1 = tk.Button(r, text='Levy Walk+Simulated Annealing', width=25, command=simulated) 
button2 = tk.Button(r, text='Levy Walk+Hill Climbing', width=25, command=rhc)
button3 = tk.Button(r, text='BackPropagation', width=25, command=backprop)
button4 = tk.Button(r, text='QUIT', width=25, command=r.destroy)
#button1.grid(row=10,column=5) 
#button2.grid(row=20,column=25) 
#button3.grid(row=30,column=45) 
w.place(x=1,y=1)
w1.place(x=590,y=51)
w1_2.place(x=560,y=101)
w2.place(x=1,y=151)
w3.place(x=50,y=251)
w4.place(x=600,y=251)
w5.place(x=1150,y=251)
button1.place(x=50,y=350)
button2.place(x=600,y=350)
button3.place(x=1150,y=350)
w6.place(x=1,y=450)
button4.place(x=600,y=600)
r.mainloop() 

'''start = datetime.now()
if(algorithm=="simulated"):
    accuracy,count,final_weights = EagleStrategyWithSA(model,20,0.2)
elif(algorithm=="rhc"):
    accuracy,count,final_weights = EagleStrategyWithHillClimbing(model,20,0.2)
print("Time taken ",datetime.now()-start)
print(accuracy,count)'''