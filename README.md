

![Alt text](https://github.com/gpls/GetAround/blob/master/get.jpg)



# ** PROJET GETAROUND  **


Deliverables : 

- ⭐️ Delay analysis dashboard in production -> https://getaround-delay-analysis-2be4960aa713.herokuapp.com/?page=home
- ⭐️ Price prediction supervised ML with MLFlow in production -> https://get-around-mlflow-54d4e77f0f13.herokuapp.com/
- ⭐️ Documented online API with /predict endpoint -> https://get-around-predict-api-922e7a48e936.herokuapp.com/


## GetAround

C'est une plateforme mettant en relation propriétaires de véhicules, particuliers comme professionnels, et conducteurs. Depuis la plateforme, le service permet aux propriétaires de partager leurs véhicules et aux conducteurs d'accéder à des véhicules en libre-service autour d'eux.

## Context

Lors de la location d'une voiture, nos utilisateurs doivent effectuer un flux d'enregistrement au début de la location et un flux de paiement à la fin de la location afin de :

Évaluez l’état de la voiture et informez les autres parties des dommages préexistants ou survenus pendant la location.
Comparez les niveaux de carburant.
Mesurez combien de kilomètres ont été parcourus.

Le check-in et le check-out de nos locations peuvent se faire avec trois flux distincts :

📱 Contrat de location mobile sur applications natives : chauffeur et propriétaire se rencontrent et signent tous deux le contrat de location sur le smartphone du propriétaire
Connecter : le conducteur ne rencontre pas le propriétaire et ouvre la voiture avec son smartphone

📝 Contrat papier (négligeable)


## Project 🚧

Lorsqu'ils utilisent Getaround, les conducteurs réservent des voitures pour une période spécifique. Les retards au retour peuvent causer des frictions pour les conducteurs suivants, car la voiture peut ne pas être disponible à temps pour leur réservation.


## Goals 🎯


Nous cherchons à mettre en place un délai minimum entre deux locations sur Getaround pour éviter les retards au retour des voitures. Nous devons déterminer le seuil de ce délai et son champ d'application. Pour prendre une décision éclairée, nous devons examiner :

- L'impact sur les revenus des propriétaires et le nombre de locations potentiellement affectées.
- La fréquence des retards au retour, leur impact sur les conducteurs suivants.
- Le nombre de cas problématiques résolus en fonction des choix de seuil et de portée.
- 

### Web dashboard

Nous allons concevoir un tableau de bord en utilisant Streamlit pour aider l'équipe de gestion de produit à analyser et répondre aux questions susmentionnées.


### Machine Learning - /predict endpoint

En plus des questions précédentes, l'équipe de Data Science utilise des données pour proposer des prix optimaux aux propriétaires de voitures à l'aide du Machine Learning.


### API
 API doit fournir au moins un point de terminaison appelé "predict". L'URL complète sera de la forme : https://votre-url.com/predict. Ce point de terminaison doit accepter la méthode POST avec des données d'entrée au format JSON, et retourner les prédictions correspondantes.


### Author

Paola Libany 