#programmeur Victor Favre
# favre.victor@protonmail.com


import pandas as pd
import numpy as np
import math as m
from math import radians, sin, cos, asin, sqrt
from datetime import datetime
import time
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from haversine import haversine, Unit
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold

# %% [markdown]
# TraiTement des données

# %% [code]
start_time = time.time()

#lecure fichier
df = pd.read_csv("train.csv")


#declaration des variables

distance=[]
temps=[]
geo4pickup=[]
geo4dropoff=[]
geo=[]
i=0
haver=[]
dayOfMonth=[]
dayOfWeek=[]
hour=[]

#filtrage des données problématique (coordonnées et temps de trajet)

df = df.drop(df[df.pickup_latitude < 40.1].index)
df = df.drop(df[df.pickup_latitude > 41.4].index)
df = df.drop(df[df.pickup_longitude >-73].index)
df = df.drop(df[df.pickup_longitude < -86].index)

df = df.drop(df[df.dropoff_latitude < 39].index)
df = df.drop(df[df.dropoff_latitude > 42].index)
df = df.drop(df[df.dropoff_longitude >-30].index)
df = df.drop(df[df.dropoff_longitude < -75].index)

df = df.drop(df[df.trip_duration < 45].index)
df = df.drop(df[df.trip_duration > 30000].index)

#iteration sur chaque ligne de la dataframe

for index, row in df.iterrows():
    datetime_pick = datetime.strptime(row["pickup_datetime"], '%Y-%m-%d %H:%M:%S')      #conversion des date en format exploitable
    datetime_drop = datetime.strptime(row["dropoff_datetime"], '%Y-%m-%d %H:%M:%S')
    delta = datetime_drop - datetime_pick
    deltaInS = delta.total_seconds()
    trip_duration = row["trip_duration"]  
    if(deltaInS != trip_duration):           #Vérification que le temps de trajet correspond entre la colonne temps de trajet et la difference entre les heure de depot et prise en charge
        df.drop(index, inplace=True)

    else:        #traitement des données
        haver.append(haversine((row["pickup_longitude"],row["pickup_latitude"]),(row["dropoff_longitude"],row["dropoff_latitude"])))  #calcul de la distance Haversine
        dayOfWeek.append(datetime_pick.weekday())
        hour.append(datetime_pick.hour)
        dayOfMonth.append(datetime_pick.day)

     # ajout de nom de chaque cluster de geohash dans une nouvelle dataframe mdf

coordspick=df.filter(["pickup_longitude","pickup_latitude"])
coordsdrop=df.filter(["dropoff_longitude","dropoff_latitude"])



# %% [code]

Sum_of_squared_distances = []
K = range(1,15)

for k in K:        # itération sur le nombre de cluster demandé a K-Means de 1 a 14
    km = KMeans(n_clusters=k)
    km = km.fit(coordsdrop)
    Sum_of_squared_distances.append(km.inertia_)
    
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k on dropoff')
plt.show()



Kpick = KMeans(init='k-means++',n_clusters=12)
Kdrop = KMeans(init='k-means++', n_clusters=14)

kmpick= Kpick.fit(coordspick)    #obtention du groupe de chaque point de l'array des coordoné en Pickup et Dropoff
kmdrop = Kdrop.fit(coordsdrop)
clusterPick=kmpick.labels_
clusterDrop=kmdrop.labels_  

mdf=df.assign(ClusterPickup=clusterPick)    #ajout de toutes les nouvelle données obtenue dans une DataFrame nommé mdf
mdf["ClusterDropoff"]=clusterDrop
mdf['DayOfWeek']=dayOfWeek
mdf['Hour']=hour
mdf['DayOfMonth']=dayOfMonth
mdf["Haversine"]=haver

# %% [code]

city_long_border = (-74.03, -73.75)     #Taille des graph pour les données
city_lat_border = (40.63, 40.85)


# graph des points des coordonées GPS pour le Pickup
cfig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(mdf.pickup_longitude.values, mdf.pickup_latitude.values, s = 0.05, alpha=0.4)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title("Pickup")
plt.show()


# graph des points des coordonées GPS pour le Dropoff

cfig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(mdf.dropoff_longitude.values, mdf.dropoff_latitude.values, s = 0.05, alpha=0.4)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title("Dropoff")
plt.show()


# graph des points des coordonées GPS pour le Pickup avec les cluster de K-means

cfig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(mdf.pickup_longitude.values, mdf.pickup_latitude.values, s=0.1, lw=0,
           c=mdf.ClusterPickup.values, cmap='tab20', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title("Pickup cluster")
plt.show()

# graph des points des coordonées GPS pour le Dropoff avec les cluster de k-means

cfig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(mdf.dropoff_longitude.values, mdf.dropoff_latitude.values, s=0.1, lw=0,
           c=mdf.ClusterDropoff.values, cmap='tab20', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title("Pickup cluster")
plt.show()

temp = mdf


#train=temp.drop(columns=["trip_duration","id",'store_and_fwd_flag','pickup_datetime',"dropoff_datetime","GeoHashPickup","GeoHashDropoff","passenger_count"])

train=temp.drop(columns=["trip_duration","id",'store_and_fwd_flag','pickup_datetime',"dropoff_datetime","passenger_count"])


#recupération de seulement les données foursissant le meilleur resultat
X=train.iloc[:, list(range(0,11))]

y=temp["trip_duration"]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=42) # sépration des données en jeu de données pour tester et entrainer


# %% [code]
#model Xgboost avec les parametres optimaux

model = XGBRegressor(
                     objective ='reg:squarederror',
                     learning_rate = 0.009,
                     n_estimators =15000,
                     min_child_weight=1,    #
                     gamma=0.8,             # parametres pour ajouter l'aspet de conversion des données entre chaque arbres
                     subsample=0.8,         #
                     colsample_bytree=0.8,  #
                     max_depth=13, # profondeur maximale de l'arbre
                     tree_method='gpu_hist',
                     gpu_id=0)

# %% [code]

eval_metric_l = ["mae","rmse"]   #Définition des differentes méthode d'evaluation des erreurs pour l'algorithme
eval_set = [(X_train, y_train), (X_test, y_test)]   
model.fit(X_train, y_train, eval_metric=eval_metric_l, eval_set=eval_set, verbose=1000, early_stopping_rounds= 45) # entrainement de l'algorithme avec un affichage de l'avancement toute les

                                                                                                                  # 1000 iterations et un arret dès une stagnation de l'avacement sur 45 rounds 



#model.fit(X_train, y_train)
y_preds=model.predict(X_test)

test=y_test.values

print("moyenne du temps de trajet = "+str((y_test.mean())))
mae=mean_absolute_error(y_preds,y_test)
rmse=mean_squared_error(y_preds,y_test)


# affichage de la RMSE et de la  MEA
print("la rmse est de : " + str(rmse))
print("la mae est de : " + str(mae))

print("accuracy base on trip average time = "+ str(1-(mae/y_test.mean())))


#affiche de la précision en %

print(1-(mae/y_test.mean()))

print("finish training")

results = model.evals_result()

x_axis = range(0,len(results['validation_0']['mae']))

plt.plot(x_axis, results['validation_0']['mae'],color="#FF2D00", label = 'Train mae')
plt.plot(x_axis, results['validation_1']['mae'],color="#924636", label = 'Test mae')
plt.legend(loc = 'best')
plt.ylabel('ite')
plt.title('Xgboost mae')
plt.show()

plt.plot(x_axis, results['validation_0']['rmse'],color="#0084F9", label = 'Train rmse')
plt.plot(x_axis, results['validation_1']['rmse'],color="#4D7599", label = 'Test rmse')
plt.ylabel('ite')
plt.title('Xgboost rmse')
plt.show() 

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.9)
plot_importance(model,max_num_features=50)
plt.show()




sdf = pd.read_csv("test.csv")
sdf.head()
#declaration des variables

haver_sub=[]
dayOfMonth_sub=[]
dayOfWeek_sub=[]
hour_sub=[]

#iteration sur chaque ligne de la dataframe

for index, row in sdf.iterrows():
    datetime_pick_sub = datetime.strptime(row["pickup_datetime"], '%Y-%m-%d %H:%M:%S')      #conversion des date en format exploitable
    haver_sub.append(haversine((row["pickup_longitude"],row["pickup_latitude"]),(row["dropoff_longitude"],row["dropoff_latitude"])))  #calcul de la distance Haversine
    dayOfWeek_sub.append(datetime_pick.weekday())
    hour_sub.append(datetime_pick.hour)
    dayOfMonth_sub.append(datetime_pick.day)


coordspick_sub=sdf.filter(["pickup_longitude","pickup_latitude"])
coordsdrop_sub=sdf.filter(["dropoff_longitude","dropoff_latitude"])
        
kmpick_sub= Kpick.fit(coordspick_sub)
kmdrop_sub = Kdrop.fit(coordsdrop_sub)
clusterPick_sub=kmpick_sub.labels_
clusterDrop_sub=kmdrop_sub.labels_  


sdf["ClusterPickup"]=clusterPick_sub
sdf["ClusterDropoff"]=clusterDrop_sub
sdf['DayOfWeek']=dayOfWeek_sub
sdf['Hour']=hour_sub
sdf['DayOfMonth']=dayOfMonth_sub
sdf["Haversine"]=haver_sub

#sdf.head()
#id=sdf["id"]
#sdf=sdf.drop(columns=["id",'store_and_fwd_flag','pickup_datetime',"passenger_count"])
#sdf.head()
#sub_preds = model.predict(sdf)


# %% [code]
sdf=sdf.drop(columns=["id",'store_and_fwd_flag','pickup_datetime',"passenger_count"])

submission_values = model.predict(sdf)

submission = pd.read_csv("sample_submission.csv")

submission["trip_duration"]= submission_values

submission.to_csv("predicitons.csv", index= False)


#programmeur Victor Favre
# favre.victor@protonmail.com