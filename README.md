[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/USx538Ll) [![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17282840&assignment_repo_type=AssignmentRepo)

# Questions&Answers

### **SETMANA 1**
- **preprocessing:**
    1. quines característiques dels datasets interessen? necessiten normalització?
    2. com es distribuiran les dades (tran-test-validació)? es treballarà amb una mostra?
    3. buscar altres datasets per utilitzar
    4. mètriques per una visió general del dataset
    5. els ratings de cada usuari estan en un rang diferent a l'average rating de les pel·licules

- **tipus de recomanador:**
    1. quins recomanadors es faran servir (*collaborative filtering, content based recommendation, ...*)?
    2. quin tipus de distància es farà servir (*Pearson, cosinus, ...*)?
    3. quines mètriques s'utilitzaràn per avaluar els resultats? quina interpretació es podrà realitzar dels resultats?

---
Treball realitzat durant setmana 1 de projecte:
**user_to_user.py:**
Hem creat i fet la primera versió de recomanació només basat en usuari per usuari. 
Explicació superficial del codi:
- Arreglar la base de dades -> pivotar per tal de tenir una matriu on files són usuaris i columnes pel·lícules. Cada cel·la correspona al rating que un usuari n ha fet d'una pel·lícula m.
- Calculem la similitud que hi ha entre els usuaris utilitzant la correlació de Pearson i la similitud del Cosinus. Per tant, creem una altra matriu: files = usuaris; columnes = usuaris; cel·les = similituds/correlacions.
- Convertim les matrius en dataframes, que ens els proporciona la llibreria de pandas per poder tractar les dades més eficientment (dataframe tracte taules de forma similar a SQL)
- Fer la funció de prediccions que consisteix en calcular les similituds amb els altres usuaris i predir quina valoració tindrà aquella pel·lícula en concret, utilitzant la fórmula mostrada a classe. 

---
---




