

![Alt text](https://github.com/gpls/GetAround/blob/master/get.jpg)




# Projet d'Analyse et de Déploiement - Getaround


Deliverables : 

- ⭐️ Delay analysis dashboard in production -> https://getaround-delay-analysis-2be4960aa713.herokuapp.com/?page=home
- ⭐️ Price prediction supervised ML with MLFlow in production -> https://get-around-mlflow-54d4e77f0f13.herokuapp.com/
- ⭐️ Documented online API with /predict endpoint -> https://get-around-predict-api-922e7a48e936.herokuapp.com/

---


## Introduction

Getaround une plateforme mettant en relation propriétaires de véhicules, particuliers comme professionnels, et conducteurs. Depuis la plateforme, le service permet aux propriétaires de partager leurs véhicules et aux conducteurs d'accéder à des véhicules en libre-service autour d'eux.

 Notre objectif est d'explorer les divers processus opérationnels liés à la location de véhicules, d'analyser l'impact des retards au retour des voitures, et de proposer des solutions pour optimiser l'expérience des utilisateurs.

 Le check-in et le check-out des locations peuvent se faire avec trois flux distincts :

📱 Contrat de location mobile sur applications natives : chauffeur et propriétaire se rencontrent et signent tous deux le contrat de location sur le smartphone du propriétaire
Connect: le conducteur ne rencontre pas le propriétaire et ouvre la voiture avec son smartphone

📝 Contrat papier (négligeable)


##  Project 🚧

### Contexte

Getaround est une plateforme innovante permettant aux conducteurs de réserver des véhicules pour une durée spécifique, offrant aux propriétaires la possibilité de partager leurs voitures. Cependant, les retards au retour des véhicules peuvent entraîner des frictions pour les conducteurs suivants, potentiellement affectant l'expérience et la satisfaction client.

### Goals 🎯

Pour répondre à ces défis, nous nous sommes fixés plusieurs objectifs clés :

1. Analyser l'impact des retards au retour sur les revenus des propriétaires et la fréquence des annulations.
2. Déterminer le seuil optimal de délai entre deux locations pour minimiser les retards, tout en évaluant son impact sur l'entreprise.
3. Implémenter des solutions basées sur l'analyse des données pour améliorer l'expérience client, tout en maintenant la rentabilité de l'entreprise.


## Contenu et Processus

### I. Exploration des Données

- **Analyse des Locations** : Étude des tendances et des grande statistiques à partir de notre base de données comprenant 8143 voitures et 21 310 locations.
- **Analyse du Pricing** : Visualisation et exploration des tarifs de location par jour, des marques les plus populaires, des types de véhicules, et des couleurs les plus recherchées.
- **Analyse des Facteurs d'Impact** : Évaluation des facteurs influençant les prix de location, tels que la marque, le type de véhicule, la couleur, et le type d'énergie.

### II. Préparation des Données

- **Nettoyage du Dataframe** : Suppression des valeurs manquantes, des colonnes inutiles, et transformation des valeurs booléennes et catégorielles en valeurs numériques.
- **Filtrage des Valeurs Aberrantes** : Utilisation de méthodes statistiques pour détecter et supprimer les valeurs aberrantes des prix de location par jour.

### III. Analyse des Retards et Impacts

- **Exploration du Délai Minimum** : Évaluation de l'impact sur les revenus des propriétaires, de la fréquence des retards, et de la résolution des cas problématiques potentiels.
- **Simulation de Scénarios** : Analyse de l'impact de différents seuils de délai entre deux locations sur les retards au retour et les revenus.

### IV. Modèles de Machine Learning et API

- **Entraînement des Modèles** : Utilisation de MlFlow pour entraîner et évaluer plusieurs modèles de machine learning de prédiction de prix de location.
- **Déploiement d'une API** : Création d'un API en ligne pour prédire les prix idéaux de location par jour en fonction des différentes caractéristiques des véhicules.

### V. Dashboard 

- **Création d'un Dashboard ** 
- Utilisation de Streamlit pour développer un tableau de bord interactif permettant d'explorer les statistiques des retards, les tendances des véhicules, et les prédictions de prix.

### VI. Déploiement et Mise en Production

- **Installation et Configuration** : Déploiement en ligne de l'application, de l'API, et des modèles de machine learning à l'aide de Heroku, FastAPI, et MlFlow.
- **Tests et Réplication** : Utilisation de Docker pour tester localement les déploiements, et mise en œuvre des corrections à partir des tests en production.

## Conclusion

Ce projet nous a permis d'appréhender de façon holistique les défis liés à la gestion des retards au retour des véhicules, et de proposer des solutions basées sur une analyse approfondie des données. Les recommandations découlant de cette analyse visent à améliorer l'expérience client tout en garantissant la réussite financière de l'entreprise.

---
### Author

Paola Libany 
