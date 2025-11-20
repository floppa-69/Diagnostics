import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EngineAnomalyDetector:
    """Real-time engine anomaly detection system"""
    
    def __init__(self, model_path='engine_anomaly_detector.tflite', 
                 scaler_path='scaler.pkl', 
                 metadata_path='model_metadata.pkl'):
        """Initialize the detector with trained model and preprocessing tools"""
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load scaler and metadata
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.feature_columns = self.metadata['feature_columns']
        self.sequence_length = self.metadata['sequence_length']
        self.class_names = self.metadata['class_names']
        
        print("✓ Model loaded successfully!")
        print(f"✓ Sequence length: {self.sequence_length}")
        print(f"✓ Features: {len(self.feature_columns)}")
        print(f"✓ Classes: {self.class_names}")
    
    def preprocess_data(self, df):
        """Preprocess raw CSV data for inference"""
        
        # Extract only the required features
        try:
            X = df[self.feature_columns].values
        except KeyError as e:
            print(f"Error: Missing required columns in CSV")
            print(f"Required: {self.feature_columns}")
            print(f"Available: {df.columns.tolist()}")
            raise e
        
        # Handle NaN and infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize using the trained scaler
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def create_sequences(self, X):
        """Create sliding window sequences"""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        return np.array(sequences, dtype=np.float32)
    
    def predict(self, sequences):
        """Run inference on sequences"""
        predictions = []
        confidences = []
        
        for sequence in sequences:
            # Prepare input
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Get prediction and confidence
            pred_class = np.argmax(output_data[0])
            confidence = output_data[0][pred_class]
            
            predictions.append(pred_class)
            confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def analyze_csv(self, csv_path):
        """Complete analysis pipeline for CSV file"""
        
        print(f"\n{'='*60}")
        print(f"ANALYZING: {csv_path}")
        print(f"{'='*60}\n")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows")
        
        # Preprocess
        X_scaled = self.preprocess_data(df)
        print(f"✓ Preprocessed data")
        
        # Create sequences
        sequences = self.create_sequences(X_scaled)
        print(f"✓ Created {len(sequences)} sequences")
        
        # Predict
        predictions, confidences = self.predict(sequences)
        print(f"✓ Completed inference")
        
        # Calculate statistics
        stats = self.calculate_statistics(predictions, confidences)
        
        # Print results
        self.print_results(stats)
        
        return predictions, confidences, stats, df
    
    def calculate_statistics(self, predictions, confidences):
        """Calculate detection statistics"""
        stats = {}
        
        # Overall statistics
        stats['total_sequences'] = len(predictions)
        stats['avg_confidence'] = np.mean(confidences)
        
        # Per-class statistics
        for i, class_name in enumerate(self.class_names):
            mask = predictions == i
            count = mask.sum()
            percentage = (count / len(predictions)) * 100
            avg_conf = confidences[mask].mean() if count > 0 else 0
            
            stats[class_name] = {
                'count': count,
                'percentage': percentage,
                'avg_confidence': avg_conf
            }
        
        # Anomaly detection (anything not "Normal")
        anomaly_mask = predictions != 0
        stats['anomalies_detected'] = anomaly_mask.sum()
        stats['anomaly_rate'] = (anomaly_mask.sum() / len(predictions)) * 100
        
        return stats
    
    def print_results(self, stats):
        """Print formatted results"""
        print(f"\n{'='*60}")
        print("DETECTION RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Total Sequences Analyzed: {stats['total_sequences']}")
        print(f"Average Confidence: {stats['avg_confidence']:.2%}")
        print(f"\nAnomaly Detection:")
        print(f"  • Anomalies Found: {stats['anomalies_detected']}")
        print(f"  • Anomaly Rate: {stats['anomaly_rate']:.2f}%")
        
        print(f"\nPer-Class Breakdown:")
        print(f"{'-'*60}")
        for class_name in self.class_names:
            class_stats = stats[class_name]
            status = "✓" if class_name == "Normal" else "⚠"
            print(f"{status} {class_name:20s}: {class_stats['count']:5d} "
                  f"({class_stats['percentage']:5.2f}%) "
                  f"[Conf: {class_stats['avg_confidence']:.2%}]")
        print(f"{'-'*60}\n")
    
    def visualize_results(self, predictions, confidences, stats, df, save_path='analysis_results.png'):
        """Create comprehensive visualization"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Time Series of Predictions
        ax1 = fig.add_subplot(gs[0, :])
        colors = ['green', 'orange', 'red', 'purple', 'blue']
        time_indices = np.arange(len(predictions))
        
        for i, class_name in enumerate(self.class_names):
            mask = predictions == i
            ax1.scatter(time_indices[mask], predictions[mask], 
                       c=colors[i], label=class_name, alpha=0.6, s=20)
        
        ax1.set_xlabel('Sequence Index', fontsize=12)
        ax1.set_ylabel('Predicted Class', fontsize=12)
        ax1.set_title('Detection Timeline', fontsize=14, fontweight='bold')
        ax1.set_yticks(range(len(self.class_names)))
        ax1.set_yticklabels(self.class_names)
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(confidences, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(confidences.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {confidences.mean():.2%}')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Class Distribution (Pie Chart)
        ax3 = fig.add_subplot(gs[1, 1])
        class_counts = [stats[name]['count'] for name in self.class_names]
        colors_pie = [colors[i] if count > 0 else 'lightgray' 
                     for i, count in enumerate(class_counts)]
        
        wedges, texts, autotexts = ax3.pie(class_counts, labels=self.class_names, 
                                            autopct='%1.1f%%', colors=colors_pie,
                                            startangle=90, textprops={'fontsize': 10})
        ax3.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # 4. Anomaly Rate Gauge
        ax4 = fig.add_subplot(gs[1, 2])
        anomaly_rate = stats['anomaly_rate']
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background arc
        ax4.plot(theta, r, 'lightgray', linewidth=20, solid_capstyle='round')
        
        # Anomaly arc
        anomaly_theta = theta[:int(len(theta) * anomaly_rate / 100)]
        color = 'green' if anomaly_rate < 5 else 'orange' if anomaly_rate < 20 else 'red'
        if len(anomaly_theta) > 0:
            ax4.plot(anomaly_theta, r[:len(anomaly_theta)], color, 
                    linewidth=20, solid_capstyle='round')
        
        ax4.text(np.pi/2, 0.5, f'{anomaly_rate:.1f}%', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax4.text(np.pi/2, 0.2, 'Anomaly Rate', 
                ha='center', va='center', fontsize=12)
        
        ax4.set_ylim(0, 1.2)
        ax4.set_xlim(0, np.pi)
        ax4.axis('off')
        ax4.set_title('Health Status', fontsize=14, fontweight='bold')
        
        # 5. Confidence by Class (Box Plot)
        ax5 = fig.add_subplot(gs[2, :2])
        conf_by_class = [confidences[predictions == i] for i in range(len(self.class_names))]
        bp = ax5.boxplot(conf_by_class, labels=self.class_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax5.set_ylabel('Confidence', fontsize=12)
        ax5.set_title('Confidence by Anomaly Type', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 6. Summary Statistics Table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_data = [
            ['Total Sequences', f"{stats['total_sequences']:,}"],
            ['Anomalies Detected', f"{stats['anomalies_detected']:,}"],
            ['Anomaly Rate', f"{stats['anomaly_rate']:.2f}%"],
            ['Avg Confidence', f"{stats['avg_confidence']:.2%}"],
            ['', ''],
            ['Engine Status', '✓ HEALTHY' if anomaly_rate < 5 else '⚠ CHECK NEEDED']
        ]
        
        table = ax6.table(cellText=summary_data, cellLoc='left',
                         bbox=[0, 0, 1, 1], colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style table
        for i, cell in table.get_celld().items():
            if i[0] == 5:  # Status row
                cell.set_facecolor('#e8f5e9' if anomaly_rate < 5 else '#fff3e0')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#f5f5f5' if i[0] % 2 == 0 else 'white')
        
        ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # Main title
        fig.suptitle('Engine Anomaly Detection Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {save_path}")
        plt.show()
    
    def generate_report(self, predictions, confidences, stats, csv_path, 
                       report_path='detection_report.txt'):
        """Generate text report"""
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ENGINE ANOMALY DETECTION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {csv_path}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Sequences Analyzed: {stats['total_sequences']:,}\n")
            f.write(f"Average Confidence: {stats['avg_confidence']:.2%}\n")
            f.write(f"Anomalies Detected: {stats['anomalies_detected']:,}\n")
            f.write(f"Anomaly Rate: {stats['anomaly_rate']:.2f}%\n\n")
            
            f.write("-"*70 + "\n")
            f.write("DETAILED BREAKDOWN\n")
            f.write("-"*70 + "\n")
            for class_name in self.class_names:
                class_stats = stats[class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Count: {class_stats['count']:,}\n")
                f.write(f"  Percentage: {class_stats['percentage']:.2f}%\n")
                f.write(f"  Avg Confidence: {class_stats['avg_confidence']:.2%}\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-"*70 + "\n")
            
            if stats['anomaly_rate'] < 5:
                f.write("✓ Engine operating normally. No action required.\n")
            elif stats['anomaly_rate'] < 20:
                f.write("⚠ Minor anomalies detected. Schedule inspection.\n")
            else:
                f.write("⚠ Significant anomalies detected. Immediate inspection recommended.\n")
            
            # Specific recommendations
            for class_name in self.class_names[1:]:  # Skip "Normal"
                if stats[class_name]['percentage'] > 5:
                    f.write(f"\n• {class_name} detected ({stats[class_name]['percentage']:.1f}%)\n")
                    if class_name == "Weak Injectors":
                        f.write("  → Check fuel injector performance and cleaning\n")
                    elif class_name == "Fuel Leak":
                        f.write("  → Inspect fuel system for leaks\n")
                    elif class_name == "Oil Leak":
                        f.write("  → Check oil levels and inspect for leaks\n")
                    elif class_name == "Vacuum Leak":
                        f.write("  → Inspect vacuum lines and intake manifold\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Report saved: {report_path}")


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("ENGINE ANOMALY DETECTION SYSTEM")
    print("Peugeot 208 2025 - 1.0L 3-Cylinder")
    print("="*70 + "\n")
    
    # Initialize detector
    try:
        detector = EngineAnomalyDetector(
            model_path='engine_anomaly_detector.tflite',
            scaler_path='scaler.pkl',
            metadata_path='model_metadata.pkl'
        )
    except FileNotFoundError as e:
        print(f"Error: Required files not found!")
        print("Please ensure these files exist:")
        print("  - engine_anomaly_detector.tflite")
        print("  - scaler.pkl")
        print("  - model_metadata.pkl")
        return
    
    # Analyze CSV file
    csv_path = input("\nEnter CSV file path (or press Enter for default): ").strip()
    if not csv_path:
        csv_path = 'file.csv'  # Your CSV file name
    
    try:
        predictions, confidences, stats, df = detector.analyze_csv(csv_path)
        
        # Visualize results
        detector.visualize_results(predictions, confidences, stats, df)
        
        # Generate report
        detector.generate_report(predictions, confidences, stats, csv_path)
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated Files:")
        print("  • analysis_results.png - Visual analysis")
        print("  • detection_report.txt - Detailed report")
        
    except FileNotFoundError:
        print(f"\nError: CSV file '{csv_path}' not found!")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
