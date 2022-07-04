# MAPPO

![MAPPO](https://github.com/annapuig/MAPPO/blob/main/Pictures/mappo.jpg)

## System Integration
### Usage of the server
In order to use the functions that runs in the server (in python), you must follow this guide.

## 1. Run the preparation.sh script


```
./preparation.sh
```

This command above will install the libraries that are needed.

## 2. Run the main

```
python3 main.py
```

This will run the app at localhost. It is useful for testing.

We access to localhost:5000, and here we can do different HTTP calls to run the different scripts.

The URLs are the following
### Routing
- Fastest Route

https://localhost:5000/routing?option=shortest&city=Vilafranca%20del%20Penedes&originx=41.33821155058145&originy=1.6916121031650182&destinationx=41.35009126561326&destinationy=1.7012326058103266&networktype=drive 

- Less Polluted Route

https://localhost:5000/routing?option=lesspolluted&city=Vilafranca%20del%20Penedes&originx=41.33821155058145&originy=1.6916121031650182&destinationx=41.35009126561326&destinationy=1.7012326058103266&networktype=drive 

- Mixed Route

https://localhost:5000/routing?option=mix&city=Vilafranca%20del%20Penedes&originx=41.33821155058145&originy=1.6916121031650182&destinationx=41.35009126561326&destinationy=1.7012326058103266&networktype=drive 

### DATABASE

- Detailed NO2

https://localhost:5000/database?option=detailed_no2

- Detailed CO

https://localhost:5000/database?option=detailed_co 

- Detailed PM10

https://localhost:5000/database?option=detailed_pm10 

- Average NO2

https://localhost:5000/database?option=avg_no2 

- Average CO

https://localhost:5000/database?option=avg_co 

- Average PM10

https://localhost:5000/database?option=avg_pm10 


