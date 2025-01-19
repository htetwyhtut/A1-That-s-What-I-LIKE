# NLP_A1_That’s What I LIKE
Enjoy reading my A1 Assignment for NLP class.

## Author Info
Name: WIN MYINT@HTET WAI YAN HTUT (WILLIAM)
Student ID: st125326

## How to run the web app
1. Pull the github repository
2. Run
```sh
python app/app.py
```
3. Access the app using http://127.0.0.1:5000

## How to use website
1. Enter Input of a single word in the search bar.
2. The program will run and display the top 10 most similar words from each model in tableformat. [ Word2Vec (Skipgram), Word2Vec Skipgram (Neg Sampling), GloVe from Scratch, and GloVe (Gensim) ]

## Training Data
1. Corpus source - nltk datasets('reuters')
2. Data source: https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html
3. Download Link in NLTK: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/reuters.zip 

## Training Parameters
1. window size   = 5
2. batch size    = 256
3. embedded size = 50
4. epoch         = 1000

## Model Comparison

| Model                | Window Size       | Training Loss   | Training Time     | Syntactic Accuracy | Semantic Accuracy |
|----------------------|-------------------|-----------------|-------------------|--------------------|-------------------|
| Skipgram             | 5                 | 17.670746       | 10 Mins 33 Secs   | 0%                 | 0%                |
| Skipgram (NEG)       | 5                 | 9.17977         | 10 Mins 48 Secs   | 0%                 | 0%                |
| Glove from Scratch   | 5                 | 597.154785      | 03 Mins 23 Secs   | 0%                 | 0%                |
| Glove (Gensim)       | 10 (Pre-trained)  | -               | -                 | 55.45%             | 93.87%            |

## Similarity Scores
| Model                | Skipgram  | NEG      | GloVe    | GloVe (Gensim) |
|----------------------|-----------|----------|----------|----------------|
| MSE                  | 32.6069   | 32.5223  | 32.6296  | 27.8081        |

## Screenshot of my web app
![Landing Page](<landing page.png>)
![Result Page](<Result of China.png>)