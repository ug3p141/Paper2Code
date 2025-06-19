"""main.py

Entry point for the CAD-Recode project.
This module reads configuration settings from config.yaml, loads or generates the dataset,
instantiates the CAD-Recode Model, coordinates the training loop using Trainer, and finally
evaluates the trained model using Evaluation. The overall flow strictly follows the design and
experimental setup described in the CAD-Recode paper.

Author: Your Name
Date: Today's Date
"""

import logging
from utils import load_config
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation

class Main:
    """Main class that coordinates dataset loading, model training, and evaluation."""
    
    def __init__(self, config: dict) -> None:
        """
        Initializes the Main class with the given configuration.
        
        Args:
            config (dict): Global configuration dictionary loaded from config.yaml.
        """
        self.config: dict = config

    def run_experiment(self) -> None:
        """Executes the full experimental pipeline: dataset loading/generation, training, and evaluation."""
        logging.info("Starting experiment using configuration:")
        logging.info(self.config)
        
        # Instantiate DatasetLoader with the configuration.
        dataset_loader = DatasetLoader(self.config)
        
        # Decide which dataset to use based on config.
        procedural_samples: int = self.config.get("dataset", {}).get("procedurally_generated_samples", 0)
        if procedural_samples > 0:
            logging.info("Generating procedural dataset with %d samples.", procedural_samples)
            dataset = dataset_loader.generate_procedural_dataset(num_samples=procedural_samples)
        else:
            alternative_dataset: str = self.config.get("dataset", {}).get("alternative_dataset", "")
            logging.info("Loading existing dataset: %s", alternative_dataset)
            dataset = dataset_loader.load_existing_dataset(dataset_name=alternative_dataset)
        
        total_samples: int = len(dataset)
        logging.info("Dataset loaded. Total samples: %d", total_samples)
        
        # Instantiate the CAD-Recode Model.
        model = Model(self.config)
        logging.info("Model instantiated.")
        
        # Instantiate Trainer with the model, dataset, and configuration.
        trainer = Trainer(model, dataset, self.config)
        logging.info("Trainer instantiated. Beginning training...")
        
        # Start training the model.
        trainer.train()
        logging.info("Training complete.")
        
        # Save model checkpoint.
        checkpoint_path: str = "checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        logging.info("Checkpoint saved at: %s", checkpoint_path)
        
        # Start evaluation of the trained model.
        logging.info("Starting evaluation...")
        evaluator = Evaluation(model, dataset, self.config)
        metrics = evaluator.evaluate()
        logging.info("Evaluation complete. Metrics:")
        logging.info(metrics)
        
        # Output the evaluation metrics.
        print("Evaluation Results:")
        print("Mean Chamfer Distance: {:.4f}".format(metrics.get("mean_CD", float('inf'))))
        print("Median Chamfer Distance: {:.4f}".format(metrics.get("median_CD", float('inf'))))
        print("Mean IoU: {:.2f}%".format(metrics.get("IoU", 0.0) * 100))
        print("Invalidity Ratio: {:.2f}%".format(metrics.get("IR", 0.0) * 100))
        print("Total samples: {}".format(metrics.get("total_samples", 0)))
        print("Valid samples: {}".format(metrics.get("valid_samples", 0)))
        

if __name__ == '__main__':
    # Load configuration from config.yaml.
    config = load_config("config.yaml")
    
    # Configure logging with a clear format and INFO level.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Create the Main instance and run the experiment.
    main_instance = Main(config)
    main_instance.run_experiment()
