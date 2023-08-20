Pour utiliser SSN au lieu de SLIC:

dans utils.py: 
    - Décommenter la ligne 13
    - Commenter lignes 46, 47, 54, 55
    - Décommenter lignes 49, 56

dans main.py:
    - Décommenter lignes 13, 14, 15

Ensuite python3 main.py normalement

Pour utiliser SH:

python3 main_SH.py
Il faut avoir au préalable les segmentations par superpixels faites sur matlab. Utiliser preprocess (dans utils.py)
pour préparer les données et utiliser le script hierarchy_preprocess.m sur MATLAB

Pour visualiser l'attention:
python3 attention.py