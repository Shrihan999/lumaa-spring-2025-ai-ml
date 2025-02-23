import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
from typing import List, Dict, Union, Optional
import logging
import yaml
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class BertEncoder:
    """
    Handles text encoding operations using BERT embeddings. This class manages the loading
    and utilization of BERT models for generating text embeddings, supporting both single
    and batch processing modes.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None):
        """
        Initializes BERT encoder with specified model and configures compute device.
        Falls back to CPU if CUDA is unavailable.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def encode_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Encodes a single text input using BERT and returns its embedding vector.
        Utilizes the [CLS] token embedding as the sequence representation.
        """
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embeddings[0]

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Performs batch encoding of multiple texts for improved performance.
        Returns a stacked array of embeddings for the entire input batch.
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
        return np.vstack(all_embeddings)

class TextPreprocessor:
    """
    Handles text preprocessing operations for movie descriptions, including tokenization,
    lemmatization, and stopword removal to standardize text input for analysis.
    """
    
    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logging.error(f"Failed to initialize NLTK components: {str(e)}")
            raise

    def get_wordnet_pos(self, word: str) -> str:
        """
        Maps POS tag to first letter of WordNet POS tags for accurate lemmatization.
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
            "J": wordnet.ADJ
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, text: str) -> str:
        """
        Performs comprehensive text preprocessing including cleaning, tokenization,
        and lemmatization with POS tagging for improved accuracy.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        
        tokens = [
            self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token))
            for token in tokens 
            if token not in self.stop_words
        ]
        
        return ' '.join(tokens)

class GenreClassifier:
    """
    Manages movie genre classification using keyword pattern matching and
    provides scoring mechanisms for genre relevance.
    """
    
    def __init__(self):
        self.genre_patterns = {
            'action': [
                r'action', r'fight', r'battle', r'explosion', r'combat', r'warrior',
                r'gun', r'war', r'mission', r'soldier', r'martial art', r'superhero',
                r'adventure', r'thriller'
            ],
            'adventure': [
                r'adventure', r'journey', r'quest', r'explore', r'discover',
                r'expedition', r'treasure', r'survival', r'fantasy', r'magical'
            ],
            'horror': [
                r'horror', r'scary', r'terror', r'frightening', r'supernatural',
                r'ghost', r'monster', r'demon', r'killer', r'slasher', r'zombie',
                r'paranormal', r'evil', r'creepy'
            ],
            'comedy': [
                r'comedy', r'funny', r'humor', r'laugh', r'hilarious', r'romantic',
                r'sitcom', r'spoof', r'parody', r'wit'
            ],
            'drama': [
                r'drama', r'emotional', r'relationship', r'life', r'family',
                r'tragic', r'powerful', r'character study', r'period piece'
            ],
            'scifi': [
                r'sci-fi', r'science fiction', r'space', r'future', r'alien',
                r'robot', r'dystopia', r'technology', r'cyberpunk'
            ]
        }
        
        self.genre_regex = {
            genre: re.compile('|'.join(patterns), re.IGNORECASE)
            for genre, patterns in self.genre_patterns.items()
        }

    def detect_genres(self, text: str) -> List[str]:
        """
        Identifies potential genres in text using predefined keyword patterns.
        """
        if not isinstance(text, str):
            return []
            
        detected_genres = []
        for genre, pattern in self.genre_regex.items():
            if pattern.search(text):
                detected_genres.append(genre)
        return detected_genres

    def get_genre_score(self, text: str, target_genre: str) -> float:
        """
        Calculates a normalized score for how well a text matches a specific genre.
        """
        if not isinstance(text, str) or not text:
            return 0.0
            
        pattern = self.genre_regex.get(target_genre.lower())
        if not pattern:
            return 0.0
            
        matches = len(pattern.findall(text.lower()))
        return min(matches / 3, 1.0)

class MovieRecommender:
    """
    Implements a BERT-based movie recommendation system with genre awareness
    and metadata processing for enhanced recommendation accuracy.
    """
    
    def __init__(self):
        self.genre_classifier = GenreClassifier()
        self.preprocessor = TextPreprocessor()
        self.bert_encoder = BertEncoder()
        self.scaler = MinMaxScaler()
        
    def _extract_genres(self, text: str) -> List[str]:
        return self.genre_classifier.detect_genres(text)

    def _create_metadata_text(self, row: pd.Series) -> str:
        """
        Generates enriched metadata text from movie attributes, with emphasis
        on genres, director, and cast information.
        """
        metadata_parts = []
        
        if pd.notna(row.get('Summary')):
            genres = self.genre_classifier.detect_genres(row['Summary'])
            metadata_parts.extend(genres * 3)
        
        if pd.notna(row.get('Director')) and isinstance(row['Director'], str):
            if len(row['Director']) < 100:
                metadata_parts.extend([row['Director']] * 2)
        
        if pd.notna(row.get('Cast')):
            cast = row['Cast'].split('|')[:3]
            metadata_parts.extend(cast)
        
        if pd.notna(row.get('Year')):
            decade = str(row['Year'])[:3] + '0s'
            metadata_parts.append(decade)
        
        return ' '.join(metadata_parts)

    def load_and_prepare_data(self, data_path: str) -> None:
        """
        Loads and prepares the movie dataset, generating BERT embeddings
        and normalizing ratings for recommendation processing.
        """
        logging.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path).sample(n=1000, random_state=42)

        
        self.data['Director'] = self.data['Director'].apply(
            lambda x: x if isinstance(x, str) and len(x) < 100 else np.nan
        )
        
        self.data['detected_genres'] = self.data['Summary'].fillna('').apply(
            self._extract_genres
        )
        
        self.data['processed_summary'] = self.data['Summary'].fillna('').apply(
            self.preprocessor.preprocess
        )
        
        self.data['metadata'] = self.data.apply(self._create_metadata_text, axis=1)
        self.data['processed_metadata'] = self.data['metadata'].apply(
            self.preprocessor.preprocess
        )
        
        logging.info("Generating BERT embeddings for summaries...")
        self.summary_embeddings = self.bert_encoder.encode_batch(
            self.data['processed_summary'].tolist()
        )
        
        logging.info("Generating BERT embeddings for metadata...")
        self.metadata_embeddings = self.bert_encoder.encode_batch(
            self.data['processed_metadata'].tolist()
        )
        
        self.data['normalized_rating'] = self.scaler.fit_transform(
            self.data['Rating'].fillna(self.data['Rating'].mean()).values.reshape(-1, 1)
        )
        
        logging.info(f"Successfully prepared {len(self.data)} movies")

    def get_recommendations(self, query: str, n: int = 5) -> List[Dict]:
        """
        Generates movie recommendations based on query text, combining BERT embeddings,
        genre matching, and rating scores for comprehensive ranking.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        target_genres = self.genre_classifier.detect_genres(query)
        if not target_genres:
            target_genres = ['drama']
            
        genre_scores = np.zeros(len(self.data))
        for genre in target_genres:
            genre_scores += self.data['Summary'].fillna('').apply(
                lambda x: self.genre_classifier.get_genre_score(x, genre)
            ).values
        
        genre_scores = genre_scores / max(len(target_genres), 1)
        
        processed_query = self.preprocessor.preprocess(query)
        query_summary_embedding = self.bert_encoder.encode_text(processed_query)
        query_metadata_embedding = self.bert_encoder.encode_text(processed_query)
        
        summary_sim = np.dot(self.summary_embeddings, query_summary_embedding) / (
            np.linalg.norm(self.summary_embeddings, axis=1) * np.linalg.norm(query_summary_embedding)
        )
        
        metadata_sim = np.dot(self.metadata_embeddings, query_metadata_embedding) / (
            np.linalg.norm(self.metadata_embeddings, axis=1) * np.linalg.norm(query_metadata_embedding)
        )
        
        final_scores = (
            0.3 * genre_scores +
            0.6 * summary_sim +
            0.2 * metadata_sim +
            0.1 * self.data['normalized_rating'].values
        )
        
        top_indices = final_scores.argsort()[-n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            movie = self.data.iloc[idx]
            movie_genres = self.genre_classifier.detect_genres(movie['Summary'])
            
            recommendations.append({
                'title': movie['Title'],
                'year': movie['Year'],
                'director': movie['Director'] if pd.notna(movie['Director']) else 'N/A',
                'rating': movie['Rating'],
                'summary': movie['Summary'][:200] + "..." if pd.notna(movie['Summary']) else '',
                'similarity_score': final_scores[idx],
                'genres': movie_genres,
                'cast': movie['Cast'].split('|')[:3] if isinstance(movie['Cast'], str) else [],
                'runtime': movie['Runtime']
            })
        
        return recommendations

def setup_logging() -> None:
    """
    Configures application logging with both console and file handlers.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('recommender.log')
        ]
    )

def main():
    """
    Main program loop that handles user interactions and generates movie recommendations.
    Sets up logging, initializes the recommender system, and manages the user input/output flow.
    """
    setup_logging()
    
    try:
        # Initialize recommender system
        recommender = MovieRecommender()
        recommender.load_and_prepare_data("data.csv")
        
        # Display welcome message and instructions
        print("\n🎬 Welcome to the BERT-Enhanced Movie Recommender! 🎬")
        print("=" * 60)
        print("Describe the type of movie you're looking for, or type 'q' to exit.")
        print("\nExample queries:")
        print("- 'Show me exciting action movies with lots of fighting'")
        print("- 'I want a scary horror movie that will frighten me'")
        print("- 'Looking for an epic adventure movie'")
        
        # Main interaction loop
        while True:
            print("\nYour movie preference:", end=" ")
            query = input().strip()
            
            # Handle exit condition
            if query.lower() == 'q':
                print("\nThanks for using Movie Recommender! Goodbye! 👋")
                break
                
            # Validate input
            if not query:
                print("Please enter a description.")
                continue
                
            # Generate and display recommendations
            print("\nFinding your perfect movies using BERT... 🔍")
            recommendations = recommender.get_recommendations(query)
            
            print("\n🎯 Top Recommendations:")
            print("-" * 60)
            
            # Format and display each recommendation
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['title']} ({rec['year']}) - ⭐ {rec['rating']:.1f}")
                print(f"   🎬 Director: {rec['director']}")
                print(f"   🎭 Cast: {', '.join(rec['cast'])}")
                print(f"   🎪 Genres: {', '.join(rec['genres'])}")
                print(f"   ⏱️  Runtime: {rec['runtime']} minutes")
                print(f"   📊 Match Score: {rec['similarity_score']:.3f}")
                print(f"   📝 Summary: {rec['summary']}")
                print()
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print("\n❌ Something went wrong. Please check the logs for details.")
        print("Make sure your dataset is properly formatted and try again.")

if __name__ == "__main__":
    main()

#Name: Shrihan Thokala
## Salary Expectation: $22/hour