# Processus complet de detection de liserés

Voici tout le processus de détection automatique des liserés contenu en un seul fichier. Il sera amélioré au fur et à mesure, mais les perofrmances dépendent surtout des réseaux de neurones utilisés, et pas vraiment du code contenu dans ce notebook.

Ce notebook utilise:

- Une grande carte (c.à.d. une image reconstruite représentant une grande surface d'échantillon à analyser)
- Un réseau de neurones

Ce notebook produit:

- Une image qui correspond à la superposition de la grande carte donnée en entrée, et des prédictions faites par le réseau de neurones sur cette carte

## Sections

1. **Paramètres**

- Un ensemble de variables à modifier pour faire configurer le processus. C'est normalement la seule partie du code que vous devrez manipuler.

2. **Imports et fonctions utilitaires**

- Importe toutes les bibliothèques nécéssaires et déclare des fonctions/classes utilitaires qui servent pour le reste du notebook

3. **Découpage de la carte**

- Permet de découper une grande carte en un jeu de données adapté au réseau de neurones. Ne doit être executé qu'une seule fois pour chaque carte.

4. **Chargement du réseau de neurones**

- Charge le réseau de neurones. Le construit automatiquement à partir de ses poids si nécéssaire.
   
5. **Prédiction sur la carte**

- Effectue l'analyse de la grande carte par le réseau de neurones

6. **Superposition et sauvegarde**

- Superpose la prédiction à la carte, et effectue la sauvegarde. Cette section peut être ré-exécutée après avoir modifié certains paramètres. 

## Comment utiliser ce notebook ?

Allez dans la section **1. Paramètres** et modifiez les paramètres appropriés, puis appuyez sur le bouton *Restart Kernel and Run all Cells* (ou quelque chose qui y ressemble, en fonction de la version de jupyter utilisée) en haut du notebook.

## Autres informations

- En python, pour commenter une ligne, on utilise le symbole *#* (équivalent au *%* de matlab)
- Pour exécuter une seule cellule, selectionnez la cellule et utilisez le raccourci **Majuscule+Entrée**. Cela fonctionne aussi avec les cellules en *markdown* comme celle qui contient ce texte.
