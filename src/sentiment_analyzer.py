from datetime import date
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from datasets import Dataset
import matplotlib.pyplot as plt
import os
from typing import Optional


class SentimentAnalyzer:
    """A general sentiment/emotion analyzer that can work with different datasets."""

    def __init__(self, input_file: str, text_column: str = 'clean_text', 
                 time_column: str = 'created_time', post_created_time: Optional[date] = None):
        self.input_file: str = input_file
        self.text_column: str = text_column
        self.time_column: str = time_column
        self.post_created_time: Optional[date] = post_created_time
        self.emotion_columns: list[str] = [
            'joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 'neutral'
        ]
        self.model_name: str = "bhadresh-savani/bert-base-uncased-emotion"
        self.emotions_data: pd.DataFrame = None
        self.data: pd.DataFrame = None 

    def load_data(self):
        """Load data from the specified input file."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"File not found: {self.input_file}")

        self.data = pd.read_csv(self.input_file)
        print(f"Loaded data from {self.input_file}: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data

    def analyze_emotions(self, output_file: str=None, batch_size: int = 128):
        """
        Analyze emotions in the text data.
        
        Args:
            output_file: Where to save results (optional)
            batch_size: How many texts to process at once
        """
         
        if self.data is None:
            self.load_data()
        
        print("Starting emotion analysis...")
        
        # Load the emotion classifier
        classifier = pipeline("text-classification", 
                            model=self.model_name, 
                            return_all_scores=True)
        
        # Analyze emotions
        dataset = Dataset.from_pandas(self.data[[self.text_column]])
        results = classifier(dataset, truncation=True, batch_size=batch_size)

        # Convert results to DataFrame
        emotion_scores = []
        for result in results:
            scores = {item['label']: item['score'] for item in result}
            emotion_scores.append(scores)
        
        emotions_df = pd.DataFrame(emotion_scores)
        
        # Combine with original data
        self.emotions_data = pd.concat([self.data, emotions_df], axis=1)
        
        # Save if requested
        if output_file:
            self.emotions_data.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        print("Emotion analysis completed!")
        return self.emotions_data
    
    def _plot_emotions(self, save_path: str = None):
        """
        Plot emotion distributions.
        
        Args:
            save_path: Path to save the plot image (optional)
        """
        plot_emotions(self.emotions_data, save_path)

    def _plot_emotion_trends(self, save_path: str = None):
        """
        Plot emotion trends over time.
        
        Args:
            save_path: Path to save the plot image (optional)
        """
        plot_emotion_trends(self.emotions_data, time_column=self.time_column, rolling_average=True, window_size=3, save_path=save_path)
    
    def _get_summary(self):
        """Get a simple summary of the emotion analysis."""
        return get_summary(self.emotions_data, emotion_columns=self.emotion_columns)


def plot_emotions(data, save_path: str = None, emotion_columns=['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 'neutral']):
    """
    Plot emotion distributions.
    
    Args:
        save_path: Path to save the plot image (optional)
    """
    if data is None:
        raise ValueError("Emotions data not available.")
    
    # Plot the distribution of each emotion

    data[emotion_columns].mean().plot(kind='bar', figsize=(10, 6), title='Average Emotion Scores')
    plt.xlabel('Emotions')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_emotion_trends(data, time_column: str = None, emotion_columns: list = None, rolling_average: bool = False, window_size: int = 3, save_path=None):
    """
    Plot emotion trends over time (if time column is available).
    
    Args:
        data: DataFrame containing emotion data
        time_column: Name of the column containing time data
        emotion_columns: List of emotion columns to plot
        window_size: Size of the rolling window for averaging (in days) 
        rolling_average: Whether to apply a rolling average to the data
        save_path: Where to save the plot (optional)
    """
    if data is None:
        print("No emotion data found. Run analyze_emotions() first.")
        return
    if emotion_columns is None or not all(col in data.columns for col in emotion_columns):
        print("No emotion columns found. Cannot plot trends.")
        return
    
    df = data.copy()
    
    # Check if time column exists
    if not time_column or time_column not in df.columns:
        print("No time column found. Cannot plot trends.")
        return
    
    # Prepare data
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)

    # Group by date and calculate daily averages
    daily_emotions = df[emotion_columns].groupby(df.index.date).agg({
        col: 'mean' for col in emotion_columns
    })

    if rolling_average:
        daily_emotions.index = pd.to_datetime(daily_emotions.index)
        daily_emotions = daily_emotions.rolling(window=f"{window_size}D").mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    # Create plot
    daily_emotions.plot(figsize=(14, 6), title=f"{window_size} Days Rolling Average Emotional Scores" if rolling_average else 'Daily Average Emotional Scores', ax=ax)

    ax.set_xlabel('Date')
    ax.set_ylabel('Emotion Score')
    ax.legend(title='Emotions')
    plt.xticks(rotation=0)
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def get_summary(data, emotion_columns=['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 'neutral']):
    """Get a simple summary of the emotion analysis."""
    if data is None:
        print("No emotion data found. Run analyze_emotions() first.")
        return
    

    if not emotion_columns:
        print("No emotion columns found.")
        return
    
    print("\n=== Emotion Analysis Summary ===")
    print(f"Total texts analyzed: {len(data)}")
    print(f"Emotions detected: {emotion_columns}")

    # Average scores
    print("\nAverage emotion scores:")
    avg_scores = data[emotion_columns].mean()
    for emotion, score in avg_scores.items():
        print(f"  {emotion}: {score:.3f}")
    
    # Most dominant emotion
    dominant_emotions = data[emotion_columns].idxmax(axis=1)
    most_common = dominant_emotions.value_counts().index[0]
    print(f"\nMost common dominant emotion: {most_common}")
    
    return avg_scores


def analyze_reddit_data(input_file='data/reddit_cleaned_final.csv', 
                       output_file='data/reddit_emotions_final.csv'):
    """Quick analysis for Reddit data."""
    analyzer = SentimentAnalyzer(
        input_file=input_file,
        text_column='clean_text',
        time_column='created_time'
    )
    
    analyzer.analyze_emotions(output_file)
    analyzer._plot_emotions('plots/reddit_emotions_simple.png')
    analyzer._plot_emotion_trends('plots/reddit_trends_simple.png')
    analyzer._get_summary()

    return analyzer


if __name__ == "__main__":
    # Analyze Reddit data
    # analyze_reddit_data(input_file='../data/reddit_cleaned_final.csv',
    #                   output_file='../data/reddit_emotions_final.csv')

    # plot reddit data
    plot_emotions(pd.read_csv('data/reddit_emotions_final.csv'), 
                  save_path='plots/reddit_emotions_final.png')
    
    argument = {
        'data': pd.read_csv('data/reddit_emotions_final.csv'),
        'time_column': 'created_time',
        'emotion_columns': ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 'neutral'],
        'rolling_average': True,
        'window_size': 3,
        'save_path': 'plots/reddit_trends_final.png'
    }

    plot_emotion_trends(data=argument['data'],
                        time_column=argument['time_column'],
                        emotion_columns=argument['emotion_columns'],
                        rolling_average=argument['rolling_average'],
                        window_size=argument['window_size'],
                        save_path=argument['save_path'])

    # replace rolling average with False to plot without rolling average from argument
    argument['rolling_average'] = False
    argument['save_path'] = 'plots/reddit_trends_final_no_rolling.png'
    argument['data'] = pd.read_csv('data/reddit_emotions_final.csv')

    plot_emotion_trends(data=argument['data'],
                        time_column=argument['time_column'],
                        emotion_columns=argument['emotion_columns'],
                        rolling_average=argument['rolling_average'],
                        window_size=argument['window_size'],
                        save_path=argument['save_path'])
    

    # Get summary
    summary = get_summary(pd.read_csv('data/reddit_emotions_final.csv'),
                            emotion_columns=['joy', 'anger', 'fear', 'sadness', 'surprise',
                                             'disgust', 'neutral'])
    print("\n=== Summary ===")
    print(summary)