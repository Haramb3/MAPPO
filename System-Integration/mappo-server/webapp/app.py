# Import OS to get the port environment variable from the Procfile
import os # <-----
import json
import requests
import osmnx as ox
import networkx as nx
import taxicab as tc
import time
import os.path
import numpy as np
import csv
import Lstm as fc
import pandas as pd
# Import the flask module
from flask import Flask, request

import mappoAPIserver_latest as api
import database as db
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from datetime import date

from itertools import islice
from pandas import DataFrame, Series, concat, read_csv, datetime

from math import sqrt
from matplotlib import pyplot
from numpy import array

from keras.backend import batch_normalization
# Create a Flask constructor. It takes name of the current module as the argument
app = Flask(__name__)
    
@app.route('/database', methods = ['GET'])

def server_database():

    option = request.args.get("option") #options are: detailed_co, detailed_no2, detailed_pm10, avg_co, avg_no2, avg_pm10
    
    return str(db.database(option))


@app.route('/routing', methods = ['GET'])

def server_less_polluted_route():
    
    option = request.args.get("option") #options are shortest, lesspolluted, mix
    city = request.args.get("city")
    originx = request.args.get("originx")
    originy = request.args.get("originy")
    destinationx = request.args.get("destinationx")
    destinationy = request.args.get("destinationy")
    networktype = request.args.get("networktype") #options are walk, drive
    #pollutionMap, pointsEdges, updated = api.filesCheck(city, networktype)
    #G = api.importFile(city, networktype)
    
    #nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    
    less_polluted, shortest, mix, lesspollutedweight, shortestweight, mixweight = api.routesComputing(originx, originy, destinationx, destinationy, city, networktype)

    
    if option == "shortest":
        #print(rutes[1])
        ruta = shortest
        
    elif option == "lesspolluted":
        #print(rutes[0])
        ruta = less_polluted
    
    elif option == "mix":
        ruta = mix    
    
    output = api.clean_output(option, ruta)
    
    
    return output




@app.route('/forecast', methods = ['GET'])

def server_forecast():
    option = request.args.get("option")
    #fc.obtain_csv(option)
    #Dataset
    dataset = pd.read_csv('avg_no2.csv', sep = ',')#index_col=1
    df = pd.DataFrame(dataset, columns=['estacio','data', 'contaminant', 'unitats', 'latitud', 'longitud', 'valor'])
    df.fillna(method="ffill", inplace=True)
    df = df.sort_values(by='data')

    #Dataset to arrays
    valores = df['valor']
    dates = df['data']
    suma = np.zeros(len(valores), float)
    dates.reset_index(drop=True, inplace=True)
    valores.reset_index(drop=True, inplace=True)
    fecha = []
    print(dates)

    #Computing daily avg 
    multiple = 8
    ultim = 943
    for i in range(len(valores)):
        if i%8 == 0 and i!=0 or i==ultim:    
            if i==ultim:
                fecha.append(dates[i-multiple+1]) 
                suma[i-multiple] = (sum(valores[(i-multiple+1):(i+1)])/multiple)
            else:
                fecha.append(dates[i-multiple])
                suma[i-multiple] = (sum(valores[(i-multiple):i])/multiple)

    avg = suma[suma!=0]
    #print((avg))
    #print(fecha)

    dfd = pd.DataFrame({'fecha': fecha, 'avg': list(avg)}, index=fecha, columns=['avg'])
    n_lag = 1
    n_seq = 1
    n_test = 18
    n_epochs = 25 #error muy grande con 200
    n_batch = 1 #idealment 512, 
    n_neurons = 10
    # prepare data
    scaler, train, test, diff_series = fc.prepare_data(dfd, n_test, n_lag, n_seq)
    # fit model
    model = fc.fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
    # make forecasts
    forecasts = fc.make_forecasts(model, n_batch, train, test, n_lag, n_seq)
    # inverse transform forecasts and test
    forecasts = fc.inverse_transform(dfd, forecasts, scaler, n_test) #n_test+2
    actual = [row[n_lag:] for row in test]
    actual = fc.inverse_transform(dfd, actual, scaler, n_test) #n_test+2
    # evaluate forecasts
    fc.evaluate_forecasts(actual, forecasts, n_lag, n_seq)

    # plot forecasts

    close_data = dfd.values.reshape((-1))
    #close_data = close_data.reshape((-1))

    look_back = 1 #1

    fechas = pd.to_datetime(fecha)
    num_prediction = 30

    forecast = fc.predict(num_prediction, model)
    # inverse transform forecasts and test
    forecasts = fc.inverse_transform_pred(dfd, forecast, scaler, num_prediction+2)
    forecast_dates = fc.predict_dates(num_prediction)

    forecasts = np.array(forecasts)
    forecasts_dates = np.array(forecast_dates)
    forecasts = forecasts.reshape(-1)

    forecasts = np.delete(forecasts, [0,1])
    forecasts_dates = np.delete(forecasts_dates, [0,1])
    forecasts_dates = pd.to_datetime(forecasts_dates)

    #Numpy 2D: 
    df_pred = pd.DataFrame({'fecha': forecasts_dates, 'prediccion': list(forecasts)}, columns = ['fecha', 'prediccion'])
    original = dfd.to_numpy
    prediction = df_pred.to_numpy

    dictionary = {'Dates': forecast_dates,
                    'Forecast': forecasts,
                    'Dates_dfd':fechas,
                    'dfd_values': dfd.values}
    
    return str(dictionary)