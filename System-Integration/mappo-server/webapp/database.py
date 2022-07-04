import psycopg2
from psycopg2 import Error

def database(option):

    #definim les queries
    detailed_no2 = """ SELECT nom_estacio, "data", contaminant, unitats, latitud, longitud, quantity
                FROM public."detailedNO2"; """
    detailed_co = """SELECT nom_estacio, "data", contaminant, unitats, latitud, longitud, quantity
                FROM public."detailedCO"; """

    detailed_pm10 = """SELECT nom_estacio, "data", contaminant, unitats, latitud, longitud, quantity
                FROM public."detailedPM10"; """
    
    avg_no2 = """SELECT public.no2_average(); """

    avg_co = """SELECT public.co_average(); """

    avg_pm10 = """SELECT public.pm10_average(); """


    try:
        # Connect to an existing database
        connection = psycopg2.connect(user="feflopfeklpznc",
                                    password="5de7e5b5fc9f83e323359f1c4ba05394ed23356dd8ba561aa45b88b54d11c026",
                                    host="ec2-54-73-167-224.eu-west-1.compute.amazonaws.com",
                                    port="5432",
                                    database="dfi7i5f0k4mkd2")

        # Create a cursor to perform database operations
        cursor = connection.cursor()
        # Print PostgreSQL details
        print("PostgreSQL server information")
        print(connection.get_dsn_parameters(), "\n")
        # Executing a SQL query

        cursor.execute(detailed_no2)
        # Fetch result
        detailed_no2 = cursor.fetchall()
        cursor.execute(detailed_co)
        # Fetch result
        detailed_co = cursor.fetchall()
        cursor.execute(detailed_pm10)
        # Fetch result
        detailed_pm10 = cursor.fetchall()
        cursor.execute(avg_no2)
        # Fetch result
        avg_no2 = cursor.fetchall()
        cursor.execute(avg_co)
        # Fetch result
        avg_co = cursor.fetchall()
        cursor.execute(avg_pm10)
        # Fetch result
        avg_pm10 = cursor.fetchall()



        if option == "detailed_no2":
            return detailed_no2
        elif option == "detailed_co":
            return detailed_co
        elif option == "detailed_pm10":
            return detailed_pm10
        elif option == "avg_no2":
            return avg_no2
        elif option == "avg_co":
            return avg_co
        elif option == "avg_pm10":
            return avg_pm10


    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

