"""
Prediction Interface for Vesta
================================
This script provides a simple interface to predict the next cycle length
given historical cycle data.
"""

import argparse

from vesta_ml.predictor import CyclePredictor


def main():
    """Command line interface for predictions."""
    parser = argparse.ArgumentParser(
        description='Predict next menstrual cycle using trained Vesta models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example CSV format (cycles.csv):
    start_date,end_date
    2023-01-15,2023-01-20
    2023-02-12,2023-02-17
    2023-03-10,2023-03-15
    ...

Example usage:
    python predict.py --input my_cycles.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to CSV file with cycle history'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VESTA CYCLE PREDICTION")
    print("="*70)
    
    # Load predictor
    print("\nLoading Random Forest model...")
    predictor = CyclePredictor(verbose=True)
    
    # Make prediction
    print(f"\nReading cycle history from: {args.input}")
    prediction = predictor.predict_from_csv(args.input)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\nModel used: {prediction['model_used'].upper()}")
    print(f"\nPredicted cycle length: {prediction['predicted_cycle_length']} days")
    print(
        "  Confidence interval: "
        f"[{prediction['confidence_interval_lower']}, {prediction['confidence_interval_upper']}] days"
    )
    
    if 'predicted_next_start_date' in prediction:
        print(f"\nPredicted next cycle start: {prediction['predicted_next_start_date']}")
        print(f"  Possible range: {prediction['date_range_lower']} to {prediction['date_range_upper']}")
    
    print("\n" + "="*70)
    
    return prediction


if __name__ == '__main__':
    main()
