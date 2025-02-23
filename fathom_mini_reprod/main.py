import torch
from .experiment import AirQuality
def main():
    # Initialize AirQuality experiment
    air_quality_experiment = AirQuality(
        batch_size=16,  # Define batch size
        num_tasks=5,  # Number of tasks (e.g., distributed clients or sub-datasets)
        window=24,  # Input window (number of time steps)
        pred_window=6,  # Prediction window (forecast horizon)
    )

    # Fetch dataset
    dataset = air_quality_experiment.get_dataset(
        percentage=0.8,  # Use 80% of the data
        device=torch.device("cpu"),  # Use CPU for the dataset
        dynamic_slide=False ,  # Enable dynamic sliding window
        concatenate=False,  # Do not concatenate multiple tasks into one dataset
        offset=0  # Start from the first client/task
    )

    # Inspect the returned dataset
    print("\n=== Dataset ===")
    print(f"Type: {type(dataset)}")  # Check the type of dataset
    print(f"Dataset Length: {len(dataset)}")  # Check how many samples/tasks are in the dataset

    # Explore a sample from the dataset
    sample = dataset[0]
    print("\n=== Sample ===")
    print(f"Sample Type: {type(sample)}")
    print(f"Sample Data: {sample}")

if __name__ == "__main__":
    main()
