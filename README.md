# Topic Modelling and Emotion Analysis of Customer Reviews

What drives negative reviews for a UK wellness centre chain? Topic modelling, emotion classification and an instruction-tuned LLM applied to reviews from two public review platforms.

**Origin:** University of Cambridge Data Science Career Accelerator coursework (2024-25). **Data:** source anonymised and not included in this repository. Notebook outputs are cleared to protect the source data; the figures below come from the original run.

## What was done
- English-language filtering, de-duplication and anonymisation (personal names and brand terms removed).
- Word frequencies and word clouds; negative reviews defined as ratings below 3.
- BERTopic (sentence-transformer embeddings) on negative reviews from locations present on both platforms.
- Location comparison: top 20 locations by negative reviews on each platform, and top 30 by total reviews.
- Emotion classification with a pre-trained BERT emotion model; BERTopic on reviews where anger was the top emotion.
- Falcon-7B-Instruct to extract three topics per review and suggest actions; Gensim LDA (10 topics) for comparison.

## Results
- 27,585 English reviews: 11,857 from Platform A and 15,728 from Platform B.
- Negative reviews: 2,422 (Platform A) and 3,396 (Platform B).
- First BERTopic run: largest themes were showers and changing-room cleanliness, then broken equipment.
- 7 locations appear in both platforms' top-20 lists for negative reviews.
- 2,031 negative reviews had anger as the top emotion (807 + 1,224). Their main themes: **cleanliness (334 reviews), staff rudeness (326), equipment (271), membership fees and cancellation (260)**.
- The LLM step produced 30 suggested actions across 10 topic clusters.

## Limitations
- The first BERTopic run used the first 1,000 negative reviews, not a random sample; Falcon was run on 50 anger reviews only.
- The emotion model labelled about 1,700 negative reviews as "joy", a likely misclassification.
- LLM-generated suggestions are generic and unvalidated.

## Tech stack
Python, pandas, NLTK, spaCy, langdetect, BERTopic, sentence-transformers, Hugging Face Transformers (BERT emotion model, Falcon-7B-Instruct), Gensim LDA, pyLDAvis, Matplotlib, WordCloud

## Repository
- `nlp_topic_modelling.ipynb`
- `requirements.txt`

---

**Author:** Raquel J. | [rjdatavoyage.co.uk](https://rjdatavoyage.co.uk)
