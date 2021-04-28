from configparser import ConfigParser

import math

import umap
import umap.plot

import numpy as np

import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt

from nn_ensemble.components.model_orchestration.model_orchestration import ModelOrchestrator
from nn_ensemble.components.data.data import Data


class Analysis:

    def __init__(self, config: ConfigParser, data: Data, model_orchestrator: ModelOrchestrator):
        """
        Args:
            config: The ConfigParser object containing the configuration.
            data: The object containing the training, validation, and testing data.
            model_orchestrator: The object containing the trained models to analyze.

        Returns:

        """
        self.config = config
        self.data = data
        self.model_orchestrator = model_orchestrator

    def analyze(self) -> None:
        """
        This function runs the analysis of the different models for
        1) Attempting to visualize mode connectivity with UMAP
        2) Attempting to visualize the functional differences between models with UMAP
        """

        self._visualize_mode_connectivity()
        self._visualize_functional_differences()

    @staticmethod
    def _extract_weight_vector(model: tf.keras.Sequential) -> np.ndarray:
        weights = model.get_weights()
        for i in range(len(weights)):
            weights[i] = weights[i].flatten()
        # weights = np.expand_dims(np.concatenate(weights))
        weights = np.concatenate(weights)
        return weights

    def _visualize_mode_connectivity(self):
        """
        This function visualizes the mode connectivity using UMAP. The hypothesis is that SWA weight should appear in
        the middle of the weight space of the individually optimized solutions.
        Returns:

        """
        # Compare UMAP embedding of weights of ensemble of models to SWA.
        models_weights = []

        model_1_ensemble_length = 0
        for model in self.model_orchestrator.load_model_generator(model_version='model_1'):
            models_weights.append(self._extract_weight_vector(model))
            model_1_ensemble_length += 1
            # Delete model instance to prevent memory issues.
            del model

        # Add the SWA model checkpoint weights for the comparison.
        model_2_n_checkpoints = 0
        for model in self.model_orchestrator.load_model_generator(model_version='model_2_checkpoints'):
            models_weights.append(self._extract_weight_vector(model))
            model_2_n_checkpoints += 1
            # Delete the model instance to prevent memory issues
            del model

        # Add the SWA model weights for the comparison.
        for model in self.model_orchestrator.load_model_generator(model_version='model_2'):
            models_weights.append(self._extract_weight_vector(model))
            # Delete model instance to prevent memory issues.
            del model

        # Concatenate all the weights into one array
        models_weights = np.array(models_weights)

        # Run the UMAP on the weight space with large n_neighbors to optimize for global structure
        mapper = umap.UMAP(n_neighbors=int(models_weights.shape[0] / 2),
                           n_components=2,
                           random_state=0,
                           metric='euclidean',
                           low_memory=True,
                           verbose=2) \
            .fit(models_weights)
        embedding = mapper.transform(models_weights)

        # plot the embedding for the ensemble
        embedding_segment = embedding[:model_1_ensemble_length, :]
        assert embedding_segment.shape[0] == model_1_ensemble_length
        s1 = plt.scatter(embedding_segment[:, 0],
                         embedding_segment[:, 1],
                         color=['red'] * model_1_ensemble_length,
                         s=0.5)

        # plot the embedding for the SWA checkpoints
        embedding_segment = embedding[model_1_ensemble_length:model_1_ensemble_length + model_2_n_checkpoints, :]
        assert embedding_segment.shape[0] == model_2_n_checkpoints
        s2 = plt.scatter(embedding_segment[:, 0],
                         embedding_segment[:, 1],
                         c=[i for i in range(model_2_n_checkpoints)],
                         cmap='Greens',
                         s=0.5)

        # plot the embedding for the SWA model
        embedding_segment = embedding[model_1_ensemble_length + model_2_n_checkpoints:, :]
        assert embedding_segment.shape[0] == 1
        s3 = plt.scatter(embedding_segment[:, 0],
                         embedding_segment[:, 1],
                         color=['blue'],
                         s=0.5)

        plt.legend((s1, s2, s3),
                   ('Ensemble Weights', 'SWA Checkpoint Weights', 'SWA Final Model Weights'),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=3,
                   fontsize=8)
        plt.title('UMAP Projection: Ensemble Weights vs. SWA Weights', fontsize=12)
        plt.tight_layout()
        plt.show()

    def _visualize_functional_differences(self):
        """
        This function visualizes the functional differences between the different model solutions by using UMAP on the
        test set prediction space.
        Returns:

        """
        num_ensemble_components_to_vis = 2

        # Custom predict loop to maintain label associations
        predictions = []
        labels = []
        sample_number = []
        iteration = 0
        for x, y in self.data.testing_dataset:
            # Compare only two models to prevent plot over crowding.
            model_generator = self.model_orchestrator.load_model_generator(model_version='model_1')
            for _ in range(num_ensemble_components_to_vis):
                model = next(model_generator)
                predictions.append(model(x, training=False).numpy())
                # Delete the model instance to clear memory
                del model
                labels.append(y.numpy())
                # Need to specify sample_number as follows due to batch processing
                sample_number.append(np.array(range(iteration, iteration + y.numpy().shape[0])))
            iteration += y.numpy().shape[0]
        predictions = np.concatenate(predictions)
        labels = np.expand_dims(np.concatenate(labels), axis=1)
        sample_number = np.expand_dims(np.concatenate(sample_number), axis=1)

        # take argmax as classification
        argmax_predictions = np.expand_dims(np.argmax(predictions, axis=-1), axis=1)

        # Add an indicator to the array for the true label and sample number.
        model_1_test_predictions = np.concatenate([predictions, argmax_predictions, labels, sample_number],
                                                  axis=1)

        # Run the UMAP on the feature prediction space (predictions are 10 dimensional vectors)
        mapper = umap.UMAP(n_neighbors=int(math.sqrt(model_1_test_predictions.shape[0])),
                           n_components=2,
                           random_state=0,
                           metric='euclidean',
                           low_memory=True,
                           verbose=2) \
            .fit(model_1_test_predictions[:int(model_1_test_predictions.shape[0]), :10])
        embedding = mapper.transform(model_1_test_predictions[:, :10])

        plt.scatter(embedding[:, 0], embedding[:, 1], c=model_1_test_predictions[:, -2], cmap='Spectral', s=0.1)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
        plt.title('UMAP Projection: Ensemble Prediction Vectors', fontsize=12)
        plt.tight_layout()
        plt.show()

        """
        Plot lines between the different model predictions on the same sample.
        While we're at it, collect some stats where the models disagree and agree.
        """
        predictions_in_agreement = []
        predictions_in_disagreement = []
        len_dataset = int(model_1_test_predictions.shape[0] / num_ensemble_components_to_vis)
        color = iter(plt.get_cmap('Spectral')(np.linspace(0, 1, len_dataset)))
        for i in range(len_dataset):
            c = next(color)
            # Mask is sample_number
            mask = model_1_test_predictions[:, -1] == i
            # Check if all models agree on outcome
            if np.all(model_1_test_predictions[mask, -3][0] == model_1_test_predictions[mask, -3]):
                predictions_in_agreement.append(model_1_test_predictions[mask, :10])
            else:
                predictions_in_disagreement.append(model_1_test_predictions[mask, :10])
            sub_embedding = embedding[mask, :]
            plt.plot(sub_embedding[:, 0], sub_embedding[:, 1], c=c, linewidth=1)
        plt.title('UMAP Projection: Ensemble Prediction Vectors Connected Samples', fontsize=12)
        plt.tight_layout()
        plt.show()

        # Convert to numpy array to take vectorized mean across sample size dimension
        predictions_in_agreement = np.concatenate(predictions_in_agreement)
        print(f'Number of predictions in agreement: {predictions_in_agreement.shape[0]}')
        print(f'Mean max prediction probability: {np.mean(np.max(predictions_in_agreement, axis=1))}')
        plt.clf()
        sns.displot(np.max(predictions_in_agreement, axis=1), kind='kde')
        plt.title('Predictions In Agreement Max Probability', fontsize=12)
        plt.tight_layout()
        plt.show()

        predictions_in_disagreement = np.concatenate(predictions_in_disagreement)
        print(f'Number of predictions in disagreement: {predictions_in_disagreement.shape[0]}')
        print(f'Mean max prediction probability: {np.mean(np.max(predictions_in_disagreement, axis=1))}')
        plt.clf()
        sns.displot(np.max(predictions_in_disagreement, axis=1), kind='kde')
        plt.title('Predictions In Disagreement Max Probability', fontsize=12)
        plt.tight_layout()
        plt.show()
