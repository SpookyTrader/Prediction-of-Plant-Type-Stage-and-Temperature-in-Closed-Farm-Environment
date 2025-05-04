import sqlite3
import pandas as pd
import numpy as np

def data_cleaning_processing():
    # Function for renaming plant type.
    def rename_plantType(x):
        if x in ['VINE CROPS','vine crops']:
            return 'Vine Crops'
        if x in ['herbs','HERBS']:
            return 'Herbs'
        if x in ['fruiting vegetables','FRUITING VEGETABLES']:
            return 'Fruiting Vegetables'
        if x in ['LEAFY GREENS','leafy greens']:
            return 'Leafy Greens'
        else:
            return x

    # Function for renaming plant stage.
    def rename_plantStage(x):
        if x in ['maturity','MATURITY']:
            return 'Maturity'
        if x in ['vegetative','VEGETATIVE']:
            return 'Vegetative'
        if x in ['seedling','SEEDLING']:
            return 'Seedling'
        else:
            return x

    # Connect and read database into pandas dataframe.
    con = sqlite3.connect("src/data/agri.db") 
    agri = pd.read_sql_query("SELECT * from farm_data", con)
    print('Extracting data...\n')
    print(agri.head())

    print('\n\nData cleaning and processing in progress...')
    
    # Apply functions to rename Plant Type and Plant Stage classes.
    agri['Plant Type'] = agri['Plant Type'].apply(rename_plantType)
    agri['Plant Stage'] = agri['Plant Stage'].apply(rename_plantStage)

    # To split the strings into number and character parts.Keep only the number parts. Convert the number parts to float.
    for f in agri.loc[:, agri.columns.str.contains('Nutrient')].columns:
        agri[f] = agri[f].apply(lambda x: float(x.split()[0]) if x is not None else x)

    agri.fillna(value=pd.NA, inplace=True)

    # Drop dulipcates and keep only the first one.
    agri.drop_duplicates(keep='first',ignore_index=True, inplace=True)

    # Replace the negative values with the medians of the variables.
    agri.loc[agri['Temperature Sensor (°C)']<0, 'Temperature Sensor (°C)'] = agri['Temperature Sensor (°C)'].median()
    agri.loc[agri['Light Intensity Sensor (lux)']<0, 'Light Intensity Sensor (lux)'] = agri['Light Intensity Sensor (lux)'].median()
    agri.loc[agri['EC Sensor (dS/m)']<0, 'EC Sensor (dS/m)'] = agri['EC Sensor (dS/m)'].median()

    # Create Plant Type-Stage feature from Plant Type and Plant Stage.
    agri['Plant Type-Stage'] = agri['Plant Type'] + '-' + agri['Plant Stage']

    # Drop rows with missing values.
    agri.dropna(inplace=True, ignore_index=True)

    print('\n\nCleaned and processed dataset:\n')
    agri.info()

    print('\nCompleted!!!')

    return agri




