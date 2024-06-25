from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import streamlit as st
import streamlit.components.v1 as components
from streamlit_vertical_slider import vertical_slider
from streamlit import session_state
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
import numpy as np
import requests


api_url = "https://get-around-predict-api-922e7a48e936.herokuapp.com/predict"

def get_page():
    return st.query_params.get('page', 'home')

def create_link(page, label):
    if page == current_page:
        return f'<a href="?page={page}" target="_top" active>{label}</a>'
    else:
        return f'<a href="?page={page}" target="_top">{label}</a>'
    
def navigate(page):
    session_state.page = page

if "page" not in session_state:
    session_state.page = "home"

current_page = get_page()
st.image("images/getaround-logo-vector.svg")
st.title("Application de location de véhicules")

st.markdown(f"""
    <div class='top-menu'>
        {create_link('home', 'Statistiques des retards')}
        {create_link('vehicule', 'Statistiques des véhicules')}
    </div>
""", unsafe_allow_html=True)

def detect_outliers(df, feature):
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    quart_rng = q3 - q1

    upper_fence = math.ceil(q3 + 1.5 * quart_rng)
    nb_rows = len(df)
    mask = (df[feature] <= upper_fence) | (df[feature].isna())
    not_outliers_count = len(df[mask])
    outliers_count = nb_rows - not_outliers_count

    return {
        'fence': upper_fence,
        'percent': round(outliers_count / nb_rows * 100, 2)
    }

def get_impact(row):
    impact = "Pas de location"
    if not math.isnan(row['checkin_delay_in_minutes']):
        if row['checkin_delay_in_minutes'] > 0:
            if row['state'] == 'Annulation':
                impact = 'Annulation'
            else:
                impact = 'Retard'
        else:
            impact = 'Aucun impact'

    return impact

def get_checkout_state(row):
    state = 'Inconnu'
    if row['state'] == 'ended':
        if row['delay_at_checkout_in_minutes'] <= 0:
            state = "Dans les temps"
        elif row['delay_at_checkout_in_minutes'] > 0:
            state = "Retard"
    if row['state'] == 'canceled':
        state = "Annulation"
    return state

def get_previous_rental_delay(row, dataframe):
    delay = np.nan
    if not math.isnan(row['previous_ended_rental_id']):
        delay = dataframe[dataframe['rental_id'] == row['previous_ended_rental_id']]['delay_at_checkout_in_minutes'].values[0]
    return delay

def create_compteur_progress(percentage, color):
    fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentage,
            gauge={
                'axis': {'range' : [0, 100]},
                'bar': {'color': color}
            },
        ))
    fig.update_layout(
        height=250,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        showlegend=False
    )
    fig.frames = [
        go.Frame(data=[go.Indicator(
            mode="gauge+number",
            value=val,
            gauge={'axis': {'range': [0, 100]},
                'bar': {'color': "lightblue"}},
        )]) for val in range(0, int(percentage) + 1)
    ]
    fig.update_layout(transition={'duration': 0})
    return fig

# Fonction pour créer une barre empilée de progression
def create_stacked_progress_bar(mobile_percentage, connect_percentage):
    fig = go.Figure(go.Bar(
        x=[mobile_percentage, connect_percentage],
        y=[''],
        orientation='h',
        text=[f'Mobile: {mobile_percentage:.2f}%', f'Connect: {connect_percentage:.2f}%'],
        textposition='inside',
        marker=dict(
            color=[st.get_option("theme.primaryColor"), 'lightgreen']
        ),
        hoverinfo='none'
    ))
    fig.update_layout(
        barmode='stack',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(showticklabels=False),
        showlegend=False,
        height=100,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def get_feature_color(feature_name):
    color_mapping = {
        'model_key': '#4287f5',
        'fuel': "#98f542",
        'car_type': "#895bfc"
    }
    for key, color in color_mapping.items():
        if feature_name.startswith(key):
            return color

    return 'gray'

def apply_time(df, time, scope):
    print(scope)
    if scope == 'Toutes les sources':
        rows = df[df['time_delta_with_previous_rental_in_minutes'] < time]
    elif scope == 'Connect':
        rows = df[(df['time_delta_with_previous_rental_in_minutes'] < time) & (df['checkin_type'] == 'connect')]
    else:
        rows = df[(df['time_delta_with_previous_rental_in_minutes'] < time) & (df['checkin_type'] == 'mobile')]

    ended_rentals = rows[(rows['state'] == 'Dans les temps') | (rows['state'] == 'Retard')]
    canceled_rentals = rows[(rows['checkin_delay_in_minutes'] > 0) & (rows['state'] == 'Annulation')]

    return (df.drop(rows.index), len(ended_rentals), len(canceled_rentals))

with open("styles.css") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

if current_page == "home":
    stats_tab, simulation_tab = st.tabs(["Statistiques", "Simulation"])

    with stats_tab:
        st.header('Statistiques sur les retards', anchor="centered-title")

        df = pd.read_csv("data/get_around_delay_analysis.csv")
        df['state'] = df.apply(get_checkout_state, axis = 1)
        
        df['previous_rental_checkout_delay_in_minutes'] = df.apply(get_previous_rental_delay, args = [df], axis = 1)
        df['checkin_delay_in_minutes'] = df['previous_rental_checkout_delay_in_minutes'] - df['time_delta_with_previous_rental_in_minutes']
        df['checkin_delay_in_minutes'] = df['checkin_delay_in_minutes'].apply(lambda x: 0 if x < 0 else x)

        reservations_annulees = df[df["state"] == "Annulation"]
        pourcentage_annulations = (len(reservations_annulees) / len(df)) * 100

        retards_livraison = df[df["delay_at_checkout_in_minutes"] > 0]
        pourcentage_retards = (len(retards_livraison) / len(df)) * 100

        retours_anticipes = df[df["delay_at_checkout_in_minutes"] < 0]
        pourcentage_anticipes = (len(retours_anticipes) / len(df)) * 100

        vehicule_count_col, location_count_col = st.columns(2)

        checkin_counts = df['checkin_type'].value_counts(normalize=True) * 100
        pourcentage_mobile = checkin_counts.get('mobile', 0)
        pourcentage_connect = checkin_counts.get('connect', 0)

        # Affichage de la distribution des checkin_type
        st.header("Répartition des réservations entre mobile et connect")
        st.plotly_chart(create_stacked_progress_bar(pourcentage_mobile, pourcentage_connect), use_container_width=True, config={'displayModeBar': False})

        with vehicule_count_col:
            st.markdown(f"Nombre de véhicules : <b>{df["car_id"].nunique()}</b>", unsafe_allow_html=True)

        with location_count_col:
            st.markdown(f"Nombre total de locations : <b>{len(df)}</b>", unsafe_allow_html=True)
        
        canceled_col, retard_col, ancitipe_col = st.columns(3)
        
        with canceled_col:
            st.header("Annulation", anchor="centered-title")
            st.plotly_chart(create_compteur_progress(pourcentage_annulations, "#4287f5"), use_container_width=True, config={'displayModeBar': False})

        with retard_col:
            st.header("Retards de livraison", anchor="centered-title")
            st.plotly_chart(create_compteur_progress(pourcentage_retards, "#ffab4b"), use_container_width=True, config={'displayModeBar': False})

        with ancitipe_col:
            st.header("Retours anticipés", anchor="centered-title")
            st.plotly_chart(create_compteur_progress(pourcentage_anticipes, "#deff4b"), use_container_width=True, config={'displayModeBar': False})

        state_counts = df['state'].value_counts()


        checkout_cols = st.columns(2)
        with checkout_cols[0]:
            # Créer un graphique circulaire avec Plotly Express
            fig = px.pie(state_counts, values=state_counts.values, names=state_counts.index,
                        title='', color=state_counts.index, color_discrete_map={
                            "Inconnu": "rgb(34, 37, 43)",
                            "Annulation": "#4287f5",
                            "Dans les temps": "#deff4b",
                            "Retard": "#ffab4b"
                        })

            # Afficher le graphique
            st.header("Répartition des états", anchor="centered-title")
            st.plotly_chart(fig, use_container_width=True)
        with checkout_cols[1]:
            late_checkouts_df = df[df['state'] == "Retard"]
            late_checkouts = detect_outliers(late_checkouts_df, 'delay_at_checkout_in_minutes')

            st.header("Retards de paiements")
            st.metric(label="", 
                value= f"{detect_outliers(df, 'delay_at_checkout_in_minutes')['percent']}%",
                delta=f"sont des valeurs aberrantes (> {late_checkouts["fence"]} minutes)",
                delta_color="inverse")

            st.metric(
                label = "", 
                value =f"{round(len(late_checkouts_df[late_checkouts_df['delay_at_checkout_in_minutes'] >= 60]) / len(late_checkouts_df) * 100, 2)}%", 
                delta = "sont d'au moins 1 heure",
                delta_color = 'inverse'
                )

            st.metric(
                    label = "", 
                    value =f"{round(len(late_checkouts_df[late_checkouts_df['delay_at_checkout_in_minutes'] <= 30]) / len(late_checkouts_df) * 100, 2)}%",
                    delta = "sont de moins de 30 min",
                    delta_color = 'normal'
                )

        df['impact_of_previous_rental_delay'] = df.apply(get_impact, axis = 1)
        previous_rental_delay_df = df[df['previous_rental_checkout_delay_in_minutes'] > 0]
        st.header("Impact des retards sur les locations suivantes")
        impact_cols = st.columns(2)
        with impact_cols[0]:
            fig = px.pie(previous_rental_delay_df, names = "impact_of_previous_rental_delay", color = "impact_of_previous_rental_delay",
                        color_discrete_map={
                            "Aucun impact": "#deff4b",
                            "Retard" : "#ffab4b",
                            "Annulation": "#4287f5",
                            "Pas de location": "rgb(34, 37, 43)"
                        })

            st.header("Répartition des types d'impact")
            st.plotly_chart(fig, use_container_width=True)
        
        with impact_cols[1]:
            late_checkins =  df[df['checkin_delay_in_minutes'] > 0]
            canceled_checkins = df[(df['checkin_delay_in_minutes'] > 0) & (df['state'] == 'Annulation')]
            st.header("En résumé...")
            st.metric(
                label = "", 
                value=f"{round(len(late_checkins) / len(df) * 100, 2)}% des retards",
                delta = "sont dû à un retard de la location précédente",
                delta_color = 'inverse')

    with simulation_tab:
        st.header("Simulation d'impact")
        st.text("Simulation de l'impact d'un temps entre chaque locations")

        with st.form(key='simulation_form'):
            input_cols = st.columns([45, 15])
            with input_cols[0]:
                time = st.slider(label='Temps entre chaque locations (en minutes)', min_value=15, max_value=721, step=15, value=15)
            with input_cols[1]:
                scope = st.radio('Réservé depuis', ['Toutes les sources', 'Connect', 'Mobile'], key=3)
            submit = st.form_submit_button(label='Voir les données')

        if submit:
            nb_ended_rentals = len(df[(df['state'] == 'Dans les temps') | (df['state'] == 'Retard')])
            nb_late_checkins = len(df[(df['checkin_delay_in_minutes'] > 0) & (df['state'] == 'Annulation')])

            with_time_df, nb_rental_lost, nb_late_checkins_cancelations_avoided = apply_time(df, time, scope)
            rental_delay_with_time = with_time_df[with_time_df["previous_rental_checkout_delay_in_minutes"] > 0]

            result_cols = st.columns(3)
            with result_cols[0]:
                st.metric("Perte de revenus", f"{round(nb_rental_lost / nb_ended_rentals * 100, 2)}%")
            with result_cols[1]:
                st.metric("Réduction du taux d'annulations", f"{round(nb_late_checkins_cancelations_avoided / nb_late_checkins * 100, 2)}%")

            before_tab, after_tab = st.tabs(["Avant", "Après"])
            with before_tab:
                fig = px.pie(previous_rental_delay_df, names = "impact_of_previous_rental_delay", color = "impact_of_previous_rental_delay",
                    color_discrete_map={
                        "Aucun impact": "#deff4b",
                        "Retard" : "#ffab4b",
                        "Annulation": "#4287f5",
                        "Pas de location": "rgb(34, 37, 43)"
                    })

                st.header("Répartition des types d'impact")
                st.plotly_chart(fig, use_container_width=True)
            
            with after_tab:
                fig = px.pie(with_time_df, names = "impact_of_previous_rental_delay", color = "impact_of_previous_rental_delay",
                    color_discrete_map={
                        "Aucun impact": "#deff4b",
                        "Retard" : "#ffab4b",
                        "Annulation": "#4287f5",
                        "Pas de location": "rgb(34, 37, 43)"
                    })

                st.header("Répartition des types d'impact")
                st.plotly_chart(fig, use_container_width=True)


elif current_page == 'vehicule':

    stats_tab, form_tab = st.tabs(["Statistiques", "Prédictions"])
    vehicule_df = pd.read_csv("data/get_around_pricing_project.csv")

    with form_tab:
        st.header("Prédiction du prix idéal de location")
        st.text("Sélectionnez les informations que vous souhaitez spécifier")

        with st.form(key="predict_form"):
            model_type_cols = st.columns([30, 30, 15, 15])
            with model_type_cols[0]:
                model_key_input = st.selectbox("Marque du véhicule", vehicule_df["model_key"].unique())
            with model_type_cols[1]:
                car_type = st.selectbox("Type de véhicule", vehicule_df["car_type"].unique())
            with model_type_cols[2]:
                color = st.selectbox("Couleur du véhicule", vehicule_df["paint_color"].unique())
            with model_type_cols[3]:
                fuel_type = st.selectbox("Énergie", vehicule_df["fuel"].unique())
            
            mileage_enpower_cols = st.columns([50, 10])
            with mileage_enpower_cols[0]:
                mileage_slider = st.slider("Nombre de kilomètres", min_value=0, max_value=2000000, value=100000, step=5000)
                options_cols = st.columns(4)
                with options_cols[0]:
                    private_parking = st.toggle("Parking privé")
                with options_cols[1]:
                    has_gps = st.toggle("Avec GPS")
                with options_cols[2]:
                    has_clim = st.toggle("Climatisation")
                with options_cols[3]:
                    is_automatic = st.toggle("Boîte automatique")

                options_cols2 = st.columns(4)
                with options_cols2[0]:
                    with_connect = st.toggle("Application Connect")
                with options_cols2[1]:
                    speed_regulator = st.toggle("Régulateur de vitesse")
                with options_cols2[2]:
                    winter_tires = st.toggle("Pneus neiges")

            with mileage_enpower_cols[1]:
                engine_power = vertical_slider("Puissance (chevaux)", min_value=40, max_value=2000, default_value=90, step=1)

            model_choice = st.selectbox("Modèle ML", ("RandomForestRegressor", "DecisionTreeRegressor", "SVR", "SGDRegressor", "HuberRegressor", "BayesianRidge", "ElasticNet", "Lars", "Lasso", "Ridge", "LinearRegression"))
            
            submit = st.form_submit_button(label="Calculer le meilleur prix")

            if submit:
                data = {
                    "model_key": model_key_input,
                    "car_type": car_type,
                    "color": color,
                    "fuel": fuel_type,
                    "mileage": mileage_slider,
                    "engine_power": engine_power,
                    "has_private_parking": private_parking,
                    "has_gps": has_gps,
                    "has_clim": has_clim,
                    "is_automatic_car": is_automatic,
                    "has_connect": with_connect,
                    "has_speed_regulator": speed_regulator,
                    "has_winter_tires": winter_tires,
                    "ml": model_choice
                }
                response = requests.post(api_url, json=data)

                if response.status_code == 200:
                    st.metric("Prix conseillé : ", round(response.json()['prediction'][0], 2))
                else:
                    st.error("Erreur lors de la prédiction")


    with stats_tab:
        st.header("Statistiques des véhicules", anchor='centered-title')


        mean_prices = vehicule_df.groupby('model_key')['rental_price_per_day'].mean().reset_index()
        st.header('Véhicules par marque')
        st.text("+ moyenne du prix de location / jour")

        # Utilisation de Plotly Express avec spécification de l'ordre des catégories
        fig = px.histogram(vehicule_df, x='model_key', y='rental_price_per_day', 
                        histfunc="avg", marginal="histogram", color_discrete_sequence=['#4287f5'],
                        category_orders={'model_key': mean_prices.sort_values(by='rental_price_per_day', ascending=False)['model_key'].tolist()},
                        labels={'model_key': 'Marque de véhicule', 'rental_price_per_day': 'Prix de location par jour'})
        fig.update_layout(yaxis_title="Prix de location par jour")
        st.plotly_chart(fig, use_container_width=True)

        fuel_cartype_cols = st.columns([25,35])
        with fuel_cartype_cols[0]:
            mean_prices = vehicule_df.groupby('fuel')['rental_price_per_day'].mean().reset_index()
            st.header("Type d'énergie")
            st.text("+ moyenne du prix de location / jour")

            # Utilisation de Plotly Express avec spécification de l'ordre des catégories
            fig = px.histogram(vehicule_df, x='fuel', y='rental_price_per_day', 
                            histfunc="avg", marginal="histogram", color_discrete_sequence=['#4287f5'],
                            category_orders={'fuel': mean_prices.sort_values(by='rental_price_per_day', ascending=False)['fuel'].tolist()},
                            labels={'fuel': "Type d'énergie", 'rental_price_per_day': 'Prix de location par jour'})
            fig.update_layout(yaxis_title="Prix de location par jour")
            st.plotly_chart(fig, use_container_width=True)

        with fuel_cartype_cols[1]:
            mean_prices = vehicule_df.groupby('car_type')['rental_price_per_day'].mean().reset_index()
            st.header("Types de véhicule")
            st.text("+ moyenne du prix de location / jour")

            # Utilisation de Plotly Express avec spécification de l'ordre des catégories
            fig = px.histogram(vehicule_df, x='car_type', y='rental_price_per_day', 
                            histfunc="avg", marginal="histogram", color_discrete_sequence=['#4287f5'],
                            category_orders={'car_type': mean_prices.sort_values(by='rental_price_per_day', ascending=False)['car_type'].tolist()},
                            labels={'car_type': 'Type de véhicule', 'rental_price_per_day': 'Prix de location par jour'})
            fig.update_layout(yaxis_title="Prix de location par jour")
            st.plotly_chart(fig, use_container_width=True)

        st.header("Couleurs des véhicules")
        st.text("+ moyenne du prix de location / jour")

        mean_prices = vehicule_df.groupby('paint_color')['rental_price_per_day'].mean().reset_index()

        fig = px.histogram(vehicule_df, x='paint_color', y='rental_price_per_day', 
                    histfunc="avg", marginal="histogram", color_discrete_sequence=['#4287f5'],
                    category_orders={'paint_color': mean_prices.sort_values(by='rental_price_per_day', ascending=False)['paint_color'].tolist()},
                    labels={'paint_color': 'Couleur de véhicule', 'rental_price_per_day': 'Prix de location par jour'})
        fig.update_layout(yaxis_title="Prix de location par jour")
        st.plotly_chart(fig, use_container_width=True)

        features = ['model_key', 'fuel', 'car_type']
        target = 'rental_price_per_day'

        # Sélection des features et de la variable cible
        features = ['model_key', 'fuel', 'car_type']
        target = 'rental_price_per_day'

        # Préparation des données
        X = vehicule_df[features]
        y = vehicule_df[target]

        # Encodage one-hot des variables catégoriques
        encoder = OneHotEncoder(drop='first')
        X_encoded = encoder.fit_transform(X)

        # Noms des colonnes après encodage
        encoded_columns = encoder.get_feature_names_out(features)

        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.header('Influence des features sur le prix de location par jour')

        st.subheader('Coefficients du modèle :')
        coefficients_df = pd.DataFrame({'feature': encoded_columns, 'coefficient': model.coef_})
        coefficients_df['color'] = coefficients_df['feature'].apply(get_feature_color)

        unique_colors = coefficients_df['color'].unique()
        
        category_order = coefficients_df.sort_values(by="coefficient", ascending=False)['feature'].tolist()

        fig_coefficients = px.bar(coefficients_df, x='feature', y='coefficient', 
                            labels={'coefficient': 'Coefficient', 'feature': 'Feature'},
                            title='Coefficients du modèle de régression linéaire',
                            color_discrete_map={color: color for color in unique_colors},
                            color='color',
                            category_orders={'feature': category_order})

        for trace, feature in zip(fig_coefficients.data, coefficients_df['feature']):
            if trace.name == "#4287f5":
                trace.name = "Marque de véhicule"
            elif trace.name == "#98f542":
                trace.name = "Type d'énergie"
            elif trace.name == "#895bfc":
                trace.name = "Type de véhicule"
    
        st.plotly_chart(fig_coefficients, use_container_width=True)