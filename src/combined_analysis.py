import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EmotionConflictAnalyzer:
    """
    A structured framework for analyzing emotions in relation to conflict events
    """
    
    def __init__(self, reddit_emotions_file, acled_file):
        self.reddit_df = pd.read_csv(reddit_emotions_file)
        self.acled_df = pd.read_csv(acled_file)
        self.combined_df = None
        self.emotion_cols = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 'neutral']
        self.significant_fatalities = 250  # Define a threshold for significant events
        self.extreme_fatalities_quantile = 0.90  # Top 5% of fatalities for extreme events

    def prepare_data(self):
        """Prepare and merge the datasets"""
        print("Preparing data...")
        
        # Prepare Reddit data
        self.reddit_df['date'] = pd.to_datetime(self.reddit_df['created_time']).dt.date
        reddit_daily = self.reddit_df.groupby('date').agg({
            **{col: 'mean' for col in self.emotion_cols},
            'comment_id': 'count'
        }).rename(columns={'comment_id': 'comment_count'})
        
        # Prepare ACLED data
        self.acled_df['date'] = pd.to_datetime(self.acled_df['event_date']).dt.date
        acled_daily = self.acled_df.groupby('date').agg(
            fatalities = ('fatalities', 'sum'),
            event_type = ('event_type', 'count'),
            civilian_targeted_events = ('civilian_targeting', lambda x: (x == 'Yes').sum())  # Count civilian targeted events
        ).rename(columns={'event_type': 'event_count'})
        
        # Merge datasets
        self.combined_df = reddit_daily.merge(acled_daily, left_index=True, right_index=True, how='left')
        self.combined_df = self.combined_df.fillna(0)
        self.combined_df.index = pd.to_datetime(self.combined_df.index)

        self.combined_df = self.combined_df.sort_index(ascending=True)
        
        print(f"Data prepared: {len(self.combined_df)} days analyzed")
        return self.combined_df
    
    def exploratory_analysis(self):
        """Basic exploratory analysis"""
        print("\nEXPLORATORY ANALYSIS")
        print("=" * 50)
        
        # Basic statistics
        print("Basic Statistics:")
        print(f"Date range: {self.combined_df.index.min()} to {self.combined_df.index.max()}")
        print(f"Total comments: {self.combined_df['comment_count'].sum():,}")
        print(f"Total events: {self.combined_df['event_count'].sum():,}")
        print(f"Total fatalities: {self.combined_df['fatalities'].sum():,}")

        # Emotion averages
        print(f"\nAverage Emotion Scores:")
        emotion_means = self.combined_df[self.emotion_cols].mean()
        for emotion, score in emotion_means.items():
            print(f"  {emotion.capitalize()}: {score:.3f}")
        
        # High-activity days
        print(f"\nHigh-Activity Days:")
        high_fatality_days = self.combined_df[self.combined_df['fatalities'] >= self.significant_fatalities]
        print(f"Days with ≥{self.significant_fatalities} fatalities: {len(high_fatality_days)}")

        if len(high_fatality_days) > 0:
            print("Top 4 deadliest days:")
            top_days = high_fatality_days.nlargest(4, 'fatalities')
            for date, row in top_days.iterrows():
                print(f"  {date.strftime('%Y-%m-%d')}: {row['fatalities']} fatalities, {row['event_count']} events")
    
    def correlation_analysis(self):
        """Analyze correlations between emotions and conflict"""
        print("\n CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Current day correlations
        correlations = self.combined_df[self.emotion_cols].corrwith(self.combined_df['fatalities'])
        print("Same-day correlations (Emotions vs Fatalities):")
        for emotion, corr in correlations.sort_values(ascending=False).items():
            significance = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
            print(f"  {emotion.capitalize()}: {corr:.3f} {significance}")
        
        # Lagged correlations (emotions responding to events)
        print(f"\nLagged correlations (Emotions vs Previous Day's Fatalities):")
        self.combined_df['fatalities_lag1'] = self.combined_df['fatalities'].shift(1)
        lag_correlations = self.combined_df[self.emotion_cols].corrwith(self.combined_df['fatalities_lag1'])
        for emotion, corr in lag_correlations.sort_values(ascending=False).items():
            significance = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
            print(f"  {emotion.capitalize()}: {corr:.3f} {significance}")
    
    def create_comprehensive_plots(self, save_dir='plots'):
        """Create comprehensive visualization suite"""
        print(f"\nCREATING VISUALIZATIONS")
        print("=" * 50)
        
        # 1. Overview timeline
        self._plot_overview_timeline(save_dir)
        
        # 2. Emotion-conflict relationships
        self._plot_emotion_conflict_relationships(save_dir)
        
        # 3. Event impact analysis
        self._plot_event_impact_analysis(save_dir)
        
        # 4. Correlation heatmap
        self._plot_correlation_heatmap(save_dir)
    
    def _plot_overview_timeline(self, save_dir):
        """Plot overview timeline"""
        fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
        
        # Negative emotions
        axes[0].plot(self.combined_df.index, self.combined_df['anger'], 'red', label='Anger', alpha=0.7)
        axes[0].plot(self.combined_df.index, self.combined_df['fear'], 'orange', label='Fear', alpha=0.7)
        axes[0].plot(self.combined_df.index, self.combined_df['sadness'], 'blue', label='Sadness', alpha=0.7)
        axes[0].set_ylabel('Negative Emotions')
        axes[0].legend()
        axes[0].set_title('Israel-Palestine Conflict: Reddit Emotions vs Events Timeline')
        
        # Positive emotions
        axes[1].plot(self.combined_df.index, self.combined_df['joy'], 'green', label='Joy', alpha=0.7)
        axes[1].plot(self.combined_df.index, self.combined_df['surprise'], 'purple', label='Surprise', alpha=0.7)
        axes[1].set_ylabel('Positive Emotions')
        axes[1].legend()
        
        # Fatalities
        axes[2].fill_between(self.combined_df.index, self.combined_df['fatalities'], 
                           color='darkred', alpha=0.6, label='Daily Fatalities')
        axes[2].set_ylabel('Fatalities')
        axes[2].legend()
        
        # Comment volume
        axes[3].fill_between(self.combined_df.index, self.combined_df['comment_count'], 
                           color='gray', alpha=0.6, label='Reddit Comments')
        axes[3].set_ylabel('Comment Count')
        axes[3].set_xlabel('Date')
        axes[3].legend()

        # Civilian harm
        axes[4].fill_between(self.combined_df.index, self.combined_df['civilian_targeted_events'], 
                           color='lightblue', alpha=0.6, label='Civilian Targeted Events')
        axes[4].set_ylabel('Civilian Targeted Events')
        axes[4].set_xlabel('Date')
        axes[4].legend()
        
        
        # Mark high-fatality days
        high_days = self.combined_df[self.combined_df['fatalities'] >= self.significant_fatalities].index
        for ax in axes:
            for i, day in enumerate(high_days):
                ax.axvline(x=day, color='red', linestyle='--', alpha=0.3, label='High Fatalities (≥250)' if i == 1 else None)

        low_targeted_days = self.combined_df[self.combined_df['civilian_targeted_events'] <=  25].index
        for ax in axes:
            for i, day in enumerate(low_targeted_days):
                ax.axvline(x=day, color='green', linestyle='--', alpha=0.3, label='Relatively Low Civilian Targeted Events (≤25)' if i == 1 else None)
        
        axes[4].legend(loc='upper right', fontsize='small')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/overview_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_emotion_conflict_relationships(self, save_dir):
        """Plot emotion-conflict relationships"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        emotions_to_plot = ['anger', 'fear', 'sadness', 'joy']
        
        for i, emotion in enumerate(emotions_to_plot):
            ax = axes[i//2, i%2]
            
            # Scatter plot
            ax.scatter(self.combined_df['fatalities'], self.combined_df[emotion], 
                      alpha=0.6, s=30)
            
            # Add trend line
            z = np.polyfit(self.combined_df['fatalities'], self.combined_df[emotion], 1)
            p = np.poly1d(z)
            ax.plot(self.combined_df['fatalities'], p(self.combined_df['fatalities']), 
                   "r--", alpha=0.8)
            
            # Calculate correlation
            corr = self.combined_df['fatalities'].corr(self.combined_df[emotion])
            
            ax.set_xlabel('Daily Fatalities')
            ax.set_ylabel(f'{emotion.capitalize()} Score')
            ax.set_title(f'{emotion.capitalize()} vs Fatalities (r={corr:.3f})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/emotion_conflict_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_event_impact_analysis(self, save_dir):
        """Analyze impact of major events"""
        # Identify major events (≥400 fatalities)
        major_events = self.combined_df[self.combined_df['fatalities'] >= self.significant_fatalities]

        if len(major_events) == 0:
            print(f"No major events (≥{self.significant_fatalities} fatalities) found in the dataset")
            return
        
        fig, axes = plt.subplots(len(major_events), 1, figsize=(15, 4*len(major_events)))
        if len(major_events) == 1:
            axes = [axes]
        
        for i, (event_date, event_data) in enumerate(major_events.iterrows()):
            ax = axes[i]
            
            # Get data for 7 days before and after
            start_date = event_date - timedelta(days=7)
            end_date = event_date + timedelta(days=7)
            
            window_data = self.combined_df[start_date:end_date]
            
            # Plot key emotions
            ax.plot(window_data.index, window_data['anger'], 'red', label='Anger', marker='o')
            ax.plot(window_data.index, window_data['fear'], 'orange', label='Fear', marker='s')
            ax.plot(window_data.index, window_data['sadness'], 'blue', label='Sadness', marker='^')
            
            # Mark the event day
            ax.axvline(x=event_date, color='black', linestyle='--', linewidth=2, 
                      label=f'Event Day ({event_data["fatalities"]} fatalities)')
            
            ax.set_title(f'Emotional Response Around Major Event: {event_date.strftime("%Y-%m-%d")}')
            ax.set_ylabel('Emotion Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/event_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_correlation_heatmap(self, save_dir):
        """Create correlation heatmap"""
        # Prepare correlation matrix
        corr_data = self.combined_df[self.emotion_cols + ['fatalities', 'comment_count']]
        correlation_matrix = corr_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.3f')
        plt.title('Correlation Matrix: Emotions vs Conflict Metrics')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def statistical_tests(self):
        """Perform statistical significance tests"""
        print(f"\nSTATISTICAL SIGNIFICANCE TESTS")
        print("=" * 50)

        # Test if emotions are different on high-fatality days take 90% of fatalities as threshold
        high_fatality_threshold = self.combined_df['fatalities'].quantile(self.extreme_fatalities_quantile)
        high_fatality_days = self.combined_df['fatalities'] >= high_fatality_threshold
        
        print(f"Comparing emotions: High-fatality days (≥{high_fatality_threshold:.1f}) vs Normal days")
        print(f"High-fatality days: {high_fatality_days.sum()}, Normal days: {(~high_fatality_days).sum()}")
        
        for emotion in self.emotion_cols:
            high_emotion = self.combined_df[high_fatality_days][emotion]
            normal_emotion = self.combined_df[~high_fatality_days][emotion]
            
            # t_stat, p_value = stats.ttest_ind(high_emotion, normal_emotion)
            u_stat, p_value = stats.mannwhitneyu(high_emotion, normal_emotion)

            
            significance = ""
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"

            print(f"  {emotion.capitalize()}: u={u_stat:.3f}, p={p_value:.4f} {significance}")

    def generate_insights(self):
        """Generate key insights from the analysis"""
        print(f"\nKEY INSIGHTS")
        print("=" * 50)
        
        # 1. Strongest emotion-conflict correlations
        correlations = self.combined_df[self.emotion_cols].corrwith(self.combined_df['fatalities'])
        strongest_emotion = correlations.abs().idxmax()
        strongest_corr = correlations[strongest_emotion]
        
        print(f"1. **{strongest_emotion.capitalize()}** shows the strongest correlation with fatalities (r={strongest_corr:.3f})")
        
        # 2. Temporal patterns
        self.combined_df['day_of_week'] = self.combined_df.index.dayofweek
        weekend_emotions = self.combined_df[self.combined_df['day_of_week'].isin([5,6])][self.emotion_cols].mean()
        weekday_emotions = self.combined_df[~self.combined_df['day_of_week'].isin([5,6])][self.emotion_cols].mean()
        
        biggest_weekend_diff = (weekend_emotions - weekday_emotions).abs().idxmax()
        print(f"2. **{biggest_weekend_diff.capitalize()}** shows the largest weekend vs weekday difference")
        
        # 3. Volume relationship
        comment_emotion_corr = self.combined_df[self.emotion_cols].corrwith(self.combined_df['comment_count'])
        volume_emotion = comment_emotion_corr.abs().idxmax()
        print(f"3. **{volume_emotion.capitalize()}** is most associated with comment volume")
        
        # 4. Event prediction potential
        lag_correlations = self.combined_df[self.emotion_cols].corrwith(self.combined_df['fatalities'].shift(-1))
        predictive_emotion = lag_correlations.abs().idxmax()
        print(f"4. **{predictive_emotion.capitalize()}** might be most predictive of next-day violence")


def run_complete_analysis():
    """Run the complete analysis pipeline"""
    print("STARTING COMPREHENSIVE EMOTION-CONFLICT ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = EmotionConflictAnalyzer(
        reddit_emotions_file='data/reddit_emotions_final.csv',
        acled_file='data/ACLED_filtered.csv'
    )
    
    # Run analysis pipeline
    analyzer.prepare_data()
    analyzer.exploratory_analysis()
    analyzer.correlation_analysis()
    analyzer.statistical_tests()
    analyzer.create_comprehensive_plots()
    analyzer.generate_insights()
    
    print("\nANALYSIS COMPLETE!")
    print("Check the 'plots' directory for visualizations.")
    
    return analyzer

# Usage
if __name__ == "__main__":
    analyzer = run_complete_analysis()