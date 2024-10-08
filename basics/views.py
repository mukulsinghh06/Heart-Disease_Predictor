from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def info(request):
    return render(request, 'info.html')

def heart(request):
    # Load and preprocess data
    path = "C:\\Users\\mukul\\OneDrive\\Desktop\\data1\\2023_24projects[1]\\2023_projects\\20_heartdiseasepredictionclassification\\heart_disease.csv"
    data = pd.read_csv(path)
    
    # Map categorical variables
    categorical_mappings = {
        'HeartDisease': {'Yes': 1, 'No': 0},
        'Smoking': {'Yes': 1, 'No': 0},
        'AlcoholDrinking': {'Yes': 1, 'No': 0},
        'Stroke': {'Yes': 1, 'No': 0},
        'DiffWalking': {'Yes': 1, 'No': 0},
        'Sex': {'Male': 1, 'Female': 0},
        'AgeCategory': {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5,
                        '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11,
                        '80 or older': 12},
        'Race': {'White': 0, 'Hispanic': 1, 'Black': 2, 'Other': 3, 'Asian': 4,
                 'American Indian/Alaskan Native': 5},
        'Diabetic': {'Yes': 1, 'No': 0, 'No, borderline diabetes': 2, 'Yes (during pregnancy)': 3},
        'PhysicalActivity': {'Yes': 1, 'No': 0},
        'GenHealth': {'Very good': 1, 'Fair': 0, 'Good': 2, 'Excellent': 3, 'Poor': 4},
        'Asthma': {'Yes': 1, 'No': 0},
        'KidneyDisease': {'Yes': 1, 'No': 0},
        'SkinCancer': {'Yes': 1, 'No': 0}
    }

    for column, mapping in categorical_mappings.items():
        data[column] = data[column].map(mapping)

    # Prepare inputs and output
    inputs = data.drop(['HeartDisease'], axis=1)
    output = data['HeartDisease']

    # Split data and scale inputs
    x_train, x_test, y_train, y_test = train_test_split(inputs, output, train_size=0.8)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Train model
    model = GaussianNB()
    model.fit(x_train, y_train)

    if request.method == "POST":
        data = request.POST
        form_fields = [
            'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth',
            'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity',
            'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer'
        ]
        
        # Collect form data
        form_data = []
        for field in form_fields:
            value = data.get(f'text{field}', '0')
            form_data.append(value)
        try:
            converted_data = [
                float(form_data[0]),  # BMI
                int(form_data[1]),    # Smoking
                int(form_data[2]),    # AlcoholDrinking
                int(form_data[3]),    # Stroke
                float(form_data[4]),  # PhysicalHealth
                float(form_data[5]),  # MentalHealth
                int(form_data[6]),    # DiffWalking
                int(form_data[7]),    # Sex
                int(form_data[8]),    # AgeCategory
                int(form_data[9]),    # Race
                int(form_data[10]),   # Diabetic
                int(form_data[11]),   # PhysicalActivity
                int(form_data[12]),   # GenHealth
                float(form_data[13]), # SleepTime
                int(form_data[14]),   # Asthma
                int(form_data[15]),   # KidneyDisease
                int(form_data[16])    # SkinCancer
            ]
            newinputs = np.array([converted_data])
        except ValueError as e:
            return render(request, 'heart.html', context={'result': f'Error: Invalid input - {str(e)}'})

        # Make prediction
        newinputs = sc.transform(newinputs)
        result = model.predict(newinputs)
        
        if result[0] == 1:
            prediction = 'The model predicts that the person has heart disease.'
        else:
            prediction = 'The model predicts that the person does not have heart disease.'
        
        return render(request, 'heart.html', context={'result': prediction})

    return render(request, 'heart.html')