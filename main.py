"""
Sentiment-Based Email Responder AI Agent
========================================

This AI agent automatically analyzes email sentiment and generates appropriate responses.
It addresses the real-world problem of efficient customer support email management.

Author: AI Assistant
Date: June 2025
"""

import pandas as pd
import numpy as np
import re
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# For sentiment analysis
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# For machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class EmailSentimentResponder:
    """
    AI Agent for sentiment-based email response generation
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.sentiment_model = LogisticRegression(random_state=42)
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Response templates based on sentiment
        self.response_templates = {
            'positive': [
                "Thank you for your positive feedback! We're delighted to hear about your experience.",
                "We appreciate your kind words and are glad we could meet your expectations.",
                "Your positive feedback motivates our team to continue delivering excellent service."
            ],
            'negative': [
                "We sincerely apologize for any inconvenience you've experienced. We take your concerns seriously.",
                "Thank you for bringing this to our attention. We're committed to resolving this issue promptly.",
                "We understand your frustration and are working to make this right for you."
            ],
            'neutral': [
                "Thank you for reaching out to us. We've received your message and will respond accordingly.",
                "We acknowledge your inquiry and will provide you with the information you need.",
                "Your message is important to us, and we'll ensure you receive proper assistance."
            ]
        }
        
        # Sample training data for demonstration
        self.sample_emails = [
            ("I love your product! It's amazing and works perfectly.", "positive"),
            ("Great customer service, very helpful staff!", "positive"),
            ("Excellent quality, highly recommend to others.", "positive"),
            ("This product is terrible, it broke after one day.", "negative"),
            ("Very disappointed with the service, will not buy again.", "negative"),
            ("Worst experience ever, completely unsatisfied.", "negative"),
            ("I need information about your return policy.", "neutral"),
            ("Can you please send me the product specifications?", "neutral"),
            ("What are your business hours?", "neutral"),
            ("The delivery was okay, product is average.", "neutral"),
            ("Absolutely fantastic service, exceeded expectations!", "positive"),
            ("Horrible quality, waste of money.", "negative")
        ]
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess email text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses, URLs, and special characters
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def analyze_sentiment_textblob(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment using TextBlob
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return "positive", polarity
        elif polarity < -0.1:
            return "negative", polarity
        else:
            return "neutral", polarity
    
    def train_model(self):
        """
        Train the sentiment classification model
        """
        print("Training sentiment analysis model...")
        
        # Prepare training data
        texts = [email[0] for email in self.sample_emails]
        labels = [email[1] for email in self.sample_emails]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(processed_texts)
        y = self.label_encoder.fit_transform(labels)
        
        # Train model
        self.sentiment_model.fit(X, y)
        
        print("Model trained successfully!")
        return True
    
    def predict_sentiment(self, email_text: str) -> Tuple[str, float]:
        """
        Predict sentiment of email using trained model
        """
        # Preprocess text
        processed_text = self.preprocess_text(email_text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.sentiment_model.predict(X)[0]
        confidence = max(self.sentiment_model.predict_proba(X)[0])
        
        # Convert back to label
        sentiment = self.label_encoder.inverse_transform([prediction])[0]
        
        return sentiment, confidence
    
    def generate_response(self, email_text: str, sender_name: str = "Customer") -> Dict:
        """
        Generate appropriate response based on email sentiment
        """
        # Get sentiment using both methods
        ml_sentiment, ml_confidence = self.predict_sentiment(email_text)
        tb_sentiment, tb_polarity = self.analyze_sentiment_textblob(email_text)
        
        # Use ML prediction as primary, TextBlob as secondary
        final_sentiment = ml_sentiment
        
        # Select appropriate response template
        import random
        response_template = random.choice(self.response_templates[final_sentiment])
        
        # Personalize response
        personalized_response = f"Dear {sender_name},\n\n{response_template}\n\nBest regards,\nCustomer Support Team"
        
        return {
            'email_text': email_text,
            'ml_sentiment': ml_sentiment,
            'ml_confidence': ml_confidence,
            'textblob_sentiment': tb_sentiment,
            'textblob_polarity': tb_polarity,
            'final_sentiment': final_sentiment,
            'generated_response': personalized_response,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def process_email_batch(self, emails: List[Dict]) -> List[Dict]:
        """
        Process multiple emails at once
        """
        results = []
        
        for email in emails:
            result = self.generate_response(
                email.get('text', ''),
                email.get('sender', 'Customer')
            )
            result['email_id'] = email.get('id', 'unknown')
            results.append(result)
        
        return results
    
    def get_sentiment_statistics(self, results: List[Dict]) -> Dict:
        """
        Generate statistics from processed emails
        """
        sentiments = [r['final_sentiment'] for r in results]
        
        stats = {
            'total_emails': len(results),
            'positive_count': sentiments.count('positive'),
            'negative_count': sentiments.count('negative'),
            'neutral_count': sentiments.count('neutral'),
            'positive_percentage': (sentiments.count('positive') / len(sentiments)) * 100,
            'negative_percentage': (sentiments.count('negative') / len(sentiments)) * 100,
            'neutral_percentage': (sentiments.count('neutral') / len(sentiments)) * 100
        }
        
        return stats


def demo_email_responder():
    """
    Demonstration of the Email Responder AI Agent
    """
    print("=" * 60)
    print("SENTIMENT-BASED EMAIL RESPONDER AI AGENT DEMO")
    print("=" * 60)
    
    # Initialize the agent
    agent = EmailSentimentResponder()
    
    # Train the model
    agent.train_model()
    print()
    
    # Sample incoming emails for demonstration
    demo_emails = [
        {
            'id': 'email_001',
            'sender': 'John Smith',
            'text': 'I absolutely love your new product! The quality is outstanding and the customer service was exceptional. Thank you so much!'
        },
        {
            'id': 'email_002',
            'sender': 'Sarah Johnson',
            'text': 'I am extremely disappointed with my recent purchase. The product arrived damaged and the return process is very complicated. This is unacceptable!'
        },
        {
            'id': 'email_003',
            'sender': 'Mike Wilson',
            'text': 'Could you please provide information about your warranty policy? I need to know the coverage details for my recent purchase.'
        },
        {
            'id': 'email_004',
            'sender': 'Emily Davis',
            'text': 'The product is okay, nothing special. Delivery was on time. Average experience overall.'
        }
    ]
    
    print("Processing incoming emails...")
    print("-" * 40)
    
    # Process emails
    results = agent.process_email_batch(demo_emails)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n📧 EMAIL {i}:")
        print(f"From: {demo_emails[i-1]['sender']}")
        print(f"Original: {result['email_text'][:100]}...")
        print(f"Sentiment: {result['final_sentiment'].upper()} (Confidence: {result['ml_confidence']:.2f})")
        print(f"TextBlob Polarity: {result['textblob_polarity']:.2f}")
        print("\n🤖 Generated Response:")
        print(result['generated_response'])
        print("-" * 40)
    
    # Show statistics
    stats = agent.get_sentiment_statistics(results)
    print(f"\n📊 PROCESSING STATISTICS:")
    print(f"Total Emails Processed: {stats['total_emails']}")
    print(f"Positive: {stats['positive_count']} ({stats['positive_percentage']:.1f}%)")
    print(f"Negative: {stats['negative_count']} ({stats['negative_percentage']:.1f}%)")
    print(f"Neutral: {stats['neutral_count']} ({stats['neutral_percentage']:.1f}%)")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return agent, results


def interactive_mode():
    """
    Interactive mode for testing the agent
    """
    print("\n🔄 INTERACTIVE MODE")
    print("Enter emails to analyze (type 'quit' to exit):")
    
    agent = EmailSentimentResponder()
    agent.train_model()
    
    while True:
        email_text = input("\nEnter email text: ")
        if email_text.lower() == 'quit':
            break
            
        sender_name = input("Enter sender name (optional): ") or "Customer"
        
        result = agent.generate_response(email_text, sender_name)
        
        print(f"\n📧 Analysis Results:")
        print(f"Sentiment: {result['final_sentiment'].upper()}")
        print(f"Confidence: {result['ml_confidence']:.2f}")
        print(f"TextBlob Polarity: {result['textblob_polarity']:.2f}")
        print(f"\n🤖 Generated Response:\n{result['generated_response']}")


if __name__ == "__main__":
    # Run demonstration
    agent, results = demo_email_responder()
    
    # Option for interactive testing
    print("\nWould you like to test the agent interactively? (y/n): ", end="")
    choice = input().lower()
    if choice == 'y':
        interactive_mode()
    
    print("\nThank you for using the Email Responder AI Agent!")
