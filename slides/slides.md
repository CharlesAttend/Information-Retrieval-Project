---
marp: true
theme: default
paginate: true
# backgroundColor: #fff
math: mathjax
# header: 
# footer: "Présentation RI"
style: |
  section.lead h1 {
    text-align: center;
  }
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
  }
---

<!-- IL MANQUE e positionnement par rapport `a l’ ́etat de l’art -->

# Injecting the BM25 Score as Text Improves BERT-Based Re-rankers

- Classiquement pipeline BM25@1000 $\rightarrow$ ReRank avec un Cross encoder (CE)
- C'est ce qui marche le mieux actuellement

---

# Cross encodeur
Architecture classique d'un cross encoder
![bg right w:100%](src/cross_encoder.png)

---

# Réutiliser le score BM25 ? 
- La méthode classique : combinaison linéaire du score CE et BM25
    - $\rightarrow$ peut réduire les performances
- :sparkles: Notre méthode :sparkles:
<!-- - Autre méthode dans "Methods for combining rankers." Est-ce qu'on en parle vite fait ? Pour moi non faudrait aller lire les papiers cités -->


---

# Méthode proposée
![bg right:40% w:100%](src/cross_encoder_bm25.png)
  Les modèle BERT savent capturer les chiffres
$\Rightarrow$ Injecter directement le score

Problèmatiques :
1. Est-ce que ça marche ? A quel point ?
2. En comparaison avec les méthodes de combinaison linéaire classique

---

<!-- 
_class: lead invert
paginate: false
-->

# Méthode

---

# Méthode : Injection du score 
<!-- Condition des expériences -->
- Le score BM25 est non borné, il faut normaliser pour garder l'interprétation
  - $Min-Max(S_{BM25}) = \frac{S_{BM25} - s_{min}}{ s_{max} - s_{min}}$
  - $Standard(S_{BM25}) = \frac{S_{BM25} - \mu(S)}{ \sigma(S) }$
  - $Sum(S_{BM25}) = \frac{S_{BM25}}{ sum(S)}$
- Deux manières de choisir les paramètres $s_{max}, s_{min}, \sigma(S), \mu(S)$
  - Local : calcule sur la "ranked list of scores per query"
  - Global : Respectivement $\{0, 50, 42, 6\}$ 
  <!-- Empirically suggested in prior work to be used as default values across different queries to globally normalize BM25 -->

---

# Méthode : Injection du score 
- Les modèles BERT ont des difficultées à interpréter les floatant
  - Conversion en nombre entier
  - Arondi des floatants

---

# Méthode : Combinaison du score BM25 et du CE
- Grosse litérature sur comment combiner les scores
- Méthode linéaire et non linéaire 
  - Somme
  - Max
  - Moyenne pondéré 
  <!-- La moyenne pondéré : poids & score CE \in [0,1], BM25 normalisé avec min max -->

  <!-- Furthermore, we train ensemble models that take sBM 25 and sCECAT as features. We experiment with three different classifiers for this purpose: SVM with a linear kernel, SVM with an RBFkernel, Naive Bayes, and Multi Layer Perceptron (MLP) as a non-linear method and report the best classifier performance in Section ???????????? -->

---

# Méthode : protocole d'évaluation

- Evaluation sur :
  - MS-Marco dev
  - TREC DL 19
  - TREC DL 20
- Métrique : 
  - nDCG@10
  - MAP
  - MRR@10 pour MS-Marco dev

---

<!-- 
_class: lead invert
paginate: false
-->

# Résultats

---

<!-- Décrire les tableau de résultat  
- T-test entre baseline et condition 
- Paramètre global > local 
  -> Les document pertinents sont boostés avec les paramètre globaux (proche de 1)
  -> En local : les documents n°1 ont toujours un score de 1 même si leurs BM25 à l'origine n'était pas très haut
- int > float 
- Min max > stadart -> avec standar il y a des score negatif
- Sum décroit les perf
- Ici la différence de perf est pas flagrante mais vous allez voir dans le tableau suivant que sur les autre dataset d'éval il y a quand même un beau boost 
-->
<!-- Dire qu'on a directement utiliser le min max global pour passer au tableau suivant -->
![center w:1000px](src/result_table_bm25_normalisation.png)

--- 

<!-- Dire qu'on a fait que les dernière ligne car 13h de train (sans early stop aie) et 40go de vram donc que la A100 de dispo -->

<!-- Décrire les tableau de résultats
- Utilisation du min max global
- Différent BERT (différence entre eux ? )
- Meilleurs résultat avec le BERT-Large
- MiniLM et autre bert résutlat similaire
-->

![center w:1000px](src/result_table_bm25_model.png)

---

<!-- Décrire les tableau de résultats
- Injection toujours mieux qu'une combinaison mathématique des deux scores
-->

![center w:1000px](src/result_table_bm25_combinaison.png)

<!-- Je skip la table suivante -->

---

<!-- Décrire les tableau de résultats
- Deux exemples de query et leur classement par un CE classique et un CE avec BM25
- Pour la première query qui est revelante ! 
  - BM25 le met en 3
  - CECAT en 1 
  - CE en 104 haiyaaa
- Pour la deuxième query qui est pas revelante 
  - BM25 le met en 146
  - CECAT en 69
  - CE en 1 haiyaaa 
- La couleurs représente the word level attribution value en utilisant l'Integrated Gradient == une valeur de d'explicabilité calculé pour BM25CAT quoi 
- On vois que le modèle utilise le score BM25 bcp pour ne pas faire ls même erreur que le CE classique
- Ils ont fait pareil sur un plus gros set de query et en triant les valeur d'importance des mots, le score BM25 est en moyenne en 3ème position => le model l'utilise beaucoup pour prendre sa décision
-->

![center w:1000px](src/result_table_query_exemple.png)

---


# Forces et faiblesses de la contribution
- Injection du score : Facile et efficace 
<!-- Dans la pipeline classique BM25@1000 et rerank -->
- Possible sur n'importe quel Cross Encoder


---

<!-- Avez-vous trouvé le papier facilement reproductible ? Quels ajustements ont  ́et ́e n cessaires ? Pour quelles raisons ? -->

# Nos remarques

- Matériel nécéssaire et temps de train des Cross Encoder assez conséquent
- AdamW à la place de Adam
- Modification de la boucle de train de Sentence Transformer pour loger les loss
- Utilisation de WandB :heart_eyes:
- Les challenges de ce projet 
  - Comprendre et utiliser les datasets
  - Trouver une mainière efficace de calculer BM25 sur le corpus
    - ~~A la main~~
    - ~~Elastic Search Docker~~
    - ~~Elastic Search Cloud~~
    - ~~pyserini sur ppti-gpu~~
    - pyserini en local $\rightarrow$ écriture du tsv $\rightarrow$ Transfert et lecture du ppti-gpu