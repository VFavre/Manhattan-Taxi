import pandas as pd
import numpy as np
import math as m
from datetime import datetime
import time
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from haversine import haversine
from sklearn.model_selection import train_test_split
from datetime import datetime

# TraiTement des données

start_time = time.time()

#lecure fichier
#df = pd.read_csv('train.csv')

#df = pd.read_csv("../input/traindata/train.csv")
df = pd.read_csv("C:\\Fast acces\\train.csv")


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

mdf = mdf.drop(columns=["id",'store_and_fwd_flag','pickup_datetime',"dropoff_datetime","passenger_count"])





sdf = pd.read_csv("C:\\Fast acces\\test.csv")
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


sdf=sdf.drop(columns=["id",'store_and_fwd_flag','pickup_datetime',"passenger_count"])


mdf.to_csv("process_Train.csv" ,index = False)

sdf.to_csv("process_Test.csv", index = False)