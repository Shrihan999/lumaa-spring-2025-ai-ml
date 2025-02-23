# BERT-Enhanced Movie Recommender

A sophisticated movie recommendation system that leverages BERT embeddings to provide intelligent movie suggestions based on natural language queries.
While the challenge suggested using TF-IDF, this implementation uses BERT for superior semantic understanding and more accurate recommendations.

## Why BERT Instead of TF-IDF?
- Better understanding of context and meaning in user queries
- Captures semantic relationships between words
- More nuanced understanding of movie descriptions
- Superior handling of word order and context
- Better match between user intent and recommendations

## Features
- BERT-based text embeddings for deep semantic understanding and the execution time is not significantly high compared to TF-IDF
- Intelligent genre classification
- Metadata-enhanced recommendation scoring
- Rating normalization
- Advanced text preprocessing
- Multi-factor scoring system combining:
  - BERT similarity (60%)
  - Metadata matching (20%)
  - Genre relevance (30%)
  - User ratings (10%)

## Dataset Requirements
The system expects a CSV file (`data.csv`) with the following columns:
- Title: Movie title
- Year: Release year
- Director: Movie director
- Cast: Pipe-separated list of cast members
- Rating: Numerical rating
- Summary: Movie plot summary
- Runtime: Movie duration in minutes

Place `data.csv` in the root directory of the project.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- CUDA-capable GPU recommended but not required (google colab T4 gpu is sufficient)

### Installation

1. Create and activate a virtual environment: (Optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Wait for BERT model download (happens automatically on first run)

## Usage

1. Run the main script:
```bash
python movie_recommender.py
```

2. Enter your movie preferences when prompted. Example queries:
```
"Show me exciting action movies with lots of fighting"
"I want a scary horror movie with supernatural elements"
"Looking for an emotional drama about family relationships"
```

## Example Output

Query: "I love thrilling action movies set in space, with a comedic twist"

```
🎯 Top Recommendations:
------------------------------------------------------------
1. Guardians of the Galaxy (2014) - ⭐ 8.0
   🎬 Director: James Gunn
   🎭 Cast: Chris Pratt, Vin Diesel, Bradley Cooper
   🎪 Genres: action, comedy, scifi
   ⏱️  Runtime: 122 minutes
   📊 Match Score: 0.892
   📝 Summary: A group of intergalactic criminals must pull together...
```

## System Architecture
- `BertEncoder`: Handles text embedding using BERT
- `TextPreprocessor`: Cleans and normalizes text data
- `GenreClassifier`: Detects movie genres using comprehensive patterns
- `MovieRecommender`: Main class that combines all components

## Performance Note
First run will be slower due to BERT model download and initialization. Subsequent runs will be faster.
I recommend to use google colab which has free T4 GPU, and it takes only 1 min to execute in google colab

## Salary Expectation
$22 per hour

## Author
Shrihan Thokala