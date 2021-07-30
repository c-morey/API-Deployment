import pandas as pd


def preprocessing():

    df = pd.read_csv('/Users/cerenmorey/Desktop/BeCode/becode_projects/Data Analysis/Final Data Analysis Project/immolisa_scrapped_data.csv')

    #Drop duplicate listings
    df.drop_duplicates(['Location', 'Property Type', 'Property Subtype', 'Price', 'Bedroom', 'Living area'],
                       keep='first', inplace=True, ignore_index=True)

    #Clean the Price column
    # First removing the dot(.)
    df['Price'] = df['Price'].str.replace('.', '')
    # Some rows have commas(,) so they need to be removed too
    df['Price'] = df['Price'].str.replace(',', '')
    # And 1 row with the 'No' value. I found the corresponding value to it from colleagues' csv.
    df['Price'].replace({'No': 275000}, inplace=True) #Replace the 'No' with the correct value
    # For this analysis, I will remove the rows that are not a number since there are only 447 counts.
    df = df[(df['Price'] != 'Make') & (df['Price'] != 'Reserve')]
    df['Price'] = df['Price'].astype(float)
    #Remove the outlier prices
    df = df[(80000 <= df.Price) & (df.Price <= 2e6)]

    #Clean the Location column, remove Nan
    df = df[df['Location'] != 0]

    #Remove grouped properties
    df = df[~df['Property Subtype'].isin(['Mixed-use building'])]

    # bedrooms should be <15
    df = df[df.Bedroom < 15]

    #Missing values for Kitchen Type
    df['Kitchen Type'] = df['Kitchen Type'].replace({'0': 'Unknown_Kitchen_Type'})

    #Missing values for Furnished
    df['Furnished'] = df['Furnished'].replace({'0': 'Unknown_Furnished'})

    #Drop the surface of the plot, it includes mix of str and int
    df = df.drop(['Surface of the plot'], axis=1)


    #Preprocessing for categorical columns
    # categoricals = []
    # for col, col_type in df.dtypes.iteritems():
    #     if col_type == 'O':
    #         categoricals.append(col)
    #     else:
    #         df[col].fillna(0, inplace=True)
    #
    # df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)


    # #Create dummy variables for text data
    type_numeric = pd.get_dummies(df['Property Type'], drop_first=False)
    subtype_numeric = pd.get_dummies(df['Property Subtype'], drop_first=False)
    province_numeric = pd.get_dummies(df['Province'], drop_first=False)
    kitchen_type_numeric = pd.get_dummies(df['Kitchen Type'], drop_first=False)
    furnished_numeric = pd.get_dummies(df['Furnished'], drop_first=False)
    fireplace_numeric = pd.get_dummies(df['HasFireplace'], drop_first=False)
    has_garden_numeric = pd.get_dummies(df['HasGarden'], drop_first=False)
    has_terrace_numeric = pd.get_dummies(df['HasTerrace'], drop_first=False)
    swimming_pool_numeric = pd.get_dummies(df['Swimming pool'], drop_first=False)
    building_condition_numeric = pd.get_dummies(df['Building condition'], drop_first=False)
    #
    # #Merge it with the df
    df = pd.concat([df, type_numeric, subtype_numeric,province_numeric,kitchen_type_numeric,furnished_numeric,fireplace_numeric,
                    has_garden_numeric,has_terrace_numeric,swimming_pool_numeric,building_condition_numeric], axis=1)

    # #Rename some columns to avoid confusion
    df.columns.values[18] = 'Type Apartment'
    df.columns.values[19] = 'Type House'

    df.columns.values[62] = 'Furnished_No'
    df.columns.values[64] = 'Furnished_Yes'

    df.columns.values[65] = 'Fireplace_No'
    df.columns.values[66] = 'Fireplace_Yes'

    df.columns.values[67] = 'HasGarden_No'
    df.columns.values[68] = 'HasGarden_Yes'

    df.columns.values[69] = 'HasTerrace_No'
    df.columns.values[70] = 'HasTerrace_Yes'

    df.columns.values[71] = 'Swimmingpool_No'
    df.columns.values[72] = 'Swimmingpool_Yes'
    #
    #
    # #Correct the data types for dummie variables
    df.iloc[:, 18:] = df.iloc[:, 18:].astype(int)


    return df

