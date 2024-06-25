from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import pandas as pd
import mlflow
import gc


mlflow.set_tracking_uri("https://get-around-mlflow-54d4e77f0f13.herokuapp.com")
mlflow.set_registry_uri("https://get-around-mlflow-54d4e77f0f13.herokuapp.com")
app = FastAPI()

class VehiculeData(BaseModel):
    model_key: str
    car_type: str
    color: str
    fuel: str
    mileage: int
    engine_power: int
    has_private_parking: bool
    has_gps: bool
    has_clim: bool
    is_automatic_car: bool
    has_connect: bool
    has_speed_regulator: bool
    has_winter_tires: bool
    ml: str

@app.post("/predict", tags=["Predictions"])
def predict(vehicle_data: VehiculeData):
    gc.collect(generation=2)

    all_runs = mlflow.search_runs(experiment_ids='0', filter_string=f"params.model = '{vehicle_data.ml}'", max_results=1)
    run_id = all_runs.iloc[0]['run_id']

    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return {"prediction": loaded_model.predict(preprocess_input(vehicle_data)).tolist()}


def preprocess_input(vehicle_data: VehiculeData) -> pd.DataFrame:
    columns = [
        'model_key_Alfa Romeo', 'model_key_Audi', 'model_key_BMW', 'model_key_Citroën', 'model_key_Ferrari',
        'model_key_Fiat', 'model_key_Ford', 'model_key_Honda', 'model_key_KIA Motors', 'model_key_Lamborghini',
        'model_key_Lexus', 'model_key_Maserati', 'model_key_Mazda', 'model_key_Mercedes', 'model_key_Mini',
        'model_key_Mitsubishi', 'model_key_Nissan', 'model_key_Opel', 'model_key_PGO', 'model_key_Peugeot',
        'model_key_Porsche', 'model_key_Renault', 'model_key_SEAT', 'model_key_Subaru', 'model_key_Suzuki',
        'model_key_Toyota', 'model_key_Volkswagen', 'model_key_Yamaha', 'fuel_diesel', 'fuel_electro',
        'fuel_hybrid_petrol', 'fuel_petrol', 'paint_color_beige', 'paint_color_black', 'paint_color_blue',
        'paint_color_brown', 'paint_color_green', 'paint_color_grey', 'paint_color_orange', 'paint_color_red',
        'paint_color_silver', 'paint_color_white', 'car_type_convertible', 'car_type_coupe', 'car_type_estate',
        'car_type_hatchback', 'car_type_sedan', 'car_type_subcompact', 'car_type_suv', 'car_type_van', 'mileage',
        'engine_power', 'private_parking_available', 'has_gps', 'has_air_conditioning', 'automatic_car',
        'has_getaround_connect', 'has_speed_regulator', 'winter_tires'
    ]
    
    input_data = pd.DataFrame(columns=columns)

    data = {
        'mileage': standardize(vehicle_data.mileage, 141004.15864491, 60162.79816124),
        'engine_power': standardize(vehicle_data.engine_power, 128.9940095, 38.92623235),
        'private_parking_available': int(vehicle_data.has_private_parking),
        'has_gps': int(vehicle_data.has_gps),
        'has_air_conditioning': int(vehicle_data.has_clim),
        'automatic_car': int(vehicle_data.is_automatic_car),
        'has_getaround_connect': int(vehicle_data.has_connect),
        'has_speed_regulator': int(vehicle_data.has_speed_regulator),
        'winter_tires': int(vehicle_data.has_winter_tires)
    }
    model_key = f'model_key_{vehicle_data.model_key}'
    if model_key in columns:
        data[model_key] = 1
    
    
    # Mappez les valeurs de carburant
    fuel_key = f'fuel_{vehicle_data.fuel}'
    if fuel_key in columns:
        data[fuel_key] = 1
    
    # Mappez les valeurs de couleur
    color_key = f'paint_color_{vehicle_data.color}'
    if color_key in columns:
        data[color_key] = 1
    
    # Mappez les valeurs de type de véhicule
    car_type_key = f'car_type_{vehicle_data.car_type}'
    if car_type_key in columns:
        data[car_type_key] = 1
    
    # Remplissez le DataFrame avec les valeurs de données
    input_data = pd.concat([input_data, pd.DataFrame([data])], ignore_index=True).fillna(0)
    
    return input_data

def standardize(value, mean, scale):
    return (value - mean) / scale