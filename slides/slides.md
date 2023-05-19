---
marp: true
theme: default 
_class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
# style: |
    img[alt~="center"] {
      display: block;
      margin: 0 auto;
    }
---

# Injecting the BM25 Score as Text Improves BERT-Based Re-rankers

- Classiquement pipeline BM25@1000 $\rightarrow$ ReRank avec un Cross encoder
[Figure 1 du papier avec le cross encoder]
- C'est ce qui marche le mieux actuellement

---
# Cross encodeur
Architecture classique d'un cross encoder
![bg right w:100%](src/cross_encoder.png)

---

# Réutiliser le score BM25 ? 
- La méthode classique : combinaison linéaire 
    - $\rightarrow$ peut réduire les performances
- :sparkles: Notre méthode :sparkles:


---
# Méthode proposée
![bg right:40% w:100%](src/cross_encoder_bm25.png)
  Les modèle BERT savent capturer les chiffres
$\Rightarrow$ Injecter directement le score

Problèmatiques :
1. Est-ce que ça marche ? A quel point ?
2. En comparaison avec les méthodes de combinaison linéaire classique

---

# Zoom sur la problématique n°1
- Les modèles BERT ont des difficultées à interpréter les floatant
- Le score BM25 est non borné, il faut normaliser

---

# Combinaison des scores
## Zoom sur la problématique n°2
- Grosse litérature sur comment combiner les scores
- Méthode linéaire et non linéaire 
- 