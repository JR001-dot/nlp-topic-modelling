# NLP Topic Modelling - Gym Customer Review Analysis

Multi-model NLP pipeline analysing customer reviews from Google and Trustpilot for a major UK gym chain, using BERTopic, LDA, emotion classification, and Falcon-7B to extract actionable business insights from unstructured text data.

## Project Overview

Understanding what drives customer satisfaction and dissatisfaction is critical for service businesses. This project applies multiple NLP approaches to thousands of gym customer reviews, systematically identifying key themes, emotional patterns, and actionable recommendations from unstructured text across two review platforms.

## Approach

**Section 1-2: Data Preparation and Investigation**
- Imported and merged 12 months of Google and Trustpilot review data
- Identified common locations across both platforms
- Language detection and filtering to English-only reviews
- Text preprocessing: tokenisation, stop word removal, custom name filtering

**Section 3: Initial Topic Modelling with BERTopic**
- BERTopic with KMeans clustering (10 topics) on negative reviews (scores < 3)
- Sentence Transformer embeddings for semantic representation
- Topic frequency analysis and document-level assignment
- Intertopic distance mapping for cluster relationships

**Section 4: Extended Investigation - Top 30 Locations**
- Location-level analysis of review volumes across both platforms
- Word frequency analysis and word cloud generation
- Comparative BERTopic run on combined reviews from high-volume locations
- Cross-comparison between negative-only and all-review topic distributions

**Section 5: Emotion Analysis with BERT**
- Hugging Face emotion classification pipeline (bhadresh-savani/bert-base-uncased-emotion)
- Six-emotion classification: anger, sadness, joy, fear, love, surprise
- Platform-level emotion distribution comparison
- Targeted BERTopic analysis on anger-classified reviews, revealing specific frustration drivers

**Section 6: Large Language Model Analysis**
- Falcon-7B-Instruct for review-level topic extraction
- Prompt engineering for structured 3-topic extraction per review
- Aggregation and clustering of LLM-extracted topics
- 10 actionable insight clusters with specific business recommendations

**Section 7: LDA Comparison with Gensim**
- Gensim LDA model (10 topics) on tokenised negative reviews
- pyLDAvis interactive visualisation of topic distances and term relevance
- Direct comparison of LDA versus BERTopic topic coherence and specificity

## Key Findings

- Cleanliness, equipment quality, staff interactions, membership processes, and overcrowding emerged as the five dominant negative themes across all methods
- Anger (approximately 2,000 reviews) and sadness (approximately 1,600) dominated negative review emotions across both platforms
- BERTopic provided more specific, actionable topics (e.g., "rude staff," "dirty showers") compared to LDA's broader thematic groupings
- 30 actionable insights generated across 10 topic clusters, with customer service training (5 clusters) and loyalty programmes (5 clusters) as highest-priority recommendations
- Location-level analysis revealed significant variation in review patterns between Google and Trustpilot for the same locations

## Tech Stack

- **NLP Models** - BERTopic, Sentence Transformers, BERT emotion classifier, Falcon-7B-Instruct
- **Topic Modelling** - BERTopic (KMeans), Gensim LDA, pyLDAvis
- **Text Processing** - NLTK (tokenisation, stop words), spaCy, regex
- **Visualisation** - Matplotlib, Seaborn, Plotly, WordCloud
- **ML/DL Frameworks** - Hugging Face Transformers, PyTorch, Scikit-learn, UMAP

## Repository Structure

```
nlp-topic-modelling/
|-- nlp_topic_modelling.ipynb   # Main analysis notebook
|-- requirements.txt            # Python dependencies
|-- README.md                   # This file
```

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook nlp_topic_modelling.ipynb
```

**Note:** The Falcon-7B model requires a GPU for reasonable inference times. The notebook was developed on Google Colab with GPU runtime.

## Dataset

12 months of customer reviews from Google Reviews and Trustpilot for a major UK gym chain, covering multiple locations across the UK. Data was sourced from the University of Cambridge Data Science programme.

## Author

**Raquel Jones** - Data Scientist and Analytics Engineer

- Portfolio: [rjdatavoyage.co.uk](https://rjdatavoyage.co.uk)
- LinkedIn: [Raquel Jones](https://linkedin.com/in/664113153)
