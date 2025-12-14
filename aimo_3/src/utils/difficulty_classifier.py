"""Problem difficulty classification system."""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Dummy classes for when sklearn is not available
    class TfidfVectorizer:
        pass
    class RandomForestClassifier:
        pass
    class LabelEncoder:
        pass


class DifficultyClassifier:
    """
    Classifies problem difficulty based on various features.
    """

    def __init__(self):
        """Initialize classifier."""
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.label_encoder = LabelEncoder()
        else:
            self.vectorizer = None
            self.classifier = None
            self.label_encoder = None
        self.is_trained = False

    def extract_features(self, problem_statement: str) -> Dict[str, float]:
        """
        Extract features from problem statement.

        Args:
            problem_statement: Problem statement in LaTeX

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Length features
        features["statement_length"] = len(problem_statement)
        features["word_count"] = len(problem_statement.split())

        # LaTeX complexity
        features["math_expressions"] = len(re.findall(r'\$[^$]+\$|\\\[.*?\\\]', problem_statement))
        features["fractions"] = len(re.findall(r'\\frac\{[^}]+\}\{[^}]+\}', problem_statement))
        features["summations"] = len(re.findall(r'\\sum', problem_statement))
        features["integrals"] = len(re.findall(r'\\int', problem_statement))
        features["limits"] = len(re.findall(r'\\lim', problem_statement))

        # Mathematical concepts
        features["has_modular_arithmetic"] = 1.0 if re.search(r'mod|remainder|divisible', problem_statement.lower()) else 0.0
        features["has_combinatorics"] = 1.0 if re.search(r'choose|combination|permutation|factorial', problem_statement.lower()) else 0.0
        features["has_geometry"] = 1.0 if re.search(r'triangle|circle|angle|area|volume', problem_statement.lower()) else 0.0
        features["has_algebra"] = 1.0 if re.search(r'equation|solve|variable|x|y|z', problem_statement.lower()) else 0.0
        features["has_number_theory"] = 1.0 if re.search(r'prime|gcd|lcm|divisor', problem_statement.lower()) else 0.0

        # Complexity indicators
        features["nested_expressions"] = problem_statement.count('{')  # Rough estimate
        features["special_functions"] = len(re.findall(r'\\log|\\sin|\\cos|\\tan|\\exp', problem_statement))

        # Normalize by length
        if features["statement_length"] > 0:
            for key in ["math_expressions", "fractions", "summations", "integrals", "limits"]:
                features[key] = features[key] / features["statement_length"]

        return features

    def train(self, problems: List[Dict], difficulty_labels: List[str]):
        """
        Train classifier on problems with difficulty labels.

        Args:
            problems: List of problem dictionaries
            difficulty_labels: List of difficulty labels ("easy", "medium", "hard")
        """
        if not SKLEARN_AVAILABLE:
            print("Warning: scikit-learn not available. Cannot train classifier.")
            return

        # Extract features
        feature_vectors = []
        text_features = []

        for problem in problems:
            statement = problem.get("statement", "")
            features = self.extract_features(statement)
            feature_vectors.append(list(features.values()))
            text_features.append(statement)

        # Combine text and feature vectors
        X_text = self.vectorizer.fit_transform(text_features).toarray()
        X_features = np.array(feature_vectors)
        X_combined = np.hstack([X_text, X_features])

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(difficulty_labels)

        # Train classifier
        self.classifier.fit(X_combined, y_encoded)
        self.is_trained = True

    def predict(self, problem_statement: str) -> Tuple[str, float]:
        """
        Predict difficulty of a problem.

        Args:
            problem_statement: Problem statement

        Returns:
            (difficulty_label, confidence) tuple
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            # Fallback to rule-based classification
            return self._rule_based_classify(problem_statement), 0.5

        # Extract features
        features = self.extract_features(problem_statement)
        feature_vector = np.array([list(features.values())])

        # Get text features
        text_vector = self.vectorizer.transform([problem_statement]).toarray()
        X_combined = np.hstack([text_vector, feature_vector])

        # Predict
        prediction = self.classifier.predict(X_combined)[0]
        probabilities = self.classifier.predict_proba(X_combined)[0]
        confidence = float(np.max(probabilities))

        difficulty = self.label_encoder.inverse_transform([prediction])[0]
        return difficulty, confidence

    def _rule_based_classify(self, problem_statement: str) -> str:
        """
        Rule-based difficulty classification (fallback).

        Args:
            problem_statement: Problem statement

        Returns:
            Difficulty label
        """
        features = self.extract_features(problem_statement)

        score = 0.0

        # Length contributes to difficulty
        if features["statement_length"] > 500:
            score += 1.0
        elif features["statement_length"] > 200:
            score += 0.5

        # Complex math operations
        if features["integrals"] > 0 or features["summations"] > 2:
            score += 2.0
        elif features["summations"] > 0 or features["limits"] > 0:
            score += 1.0

        # Multiple concepts
        concept_count = sum([
            features["has_modular_arithmetic"],
            features["has_combinatorics"],
            features["has_geometry"],
            features["has_algebra"],
            features["has_number_theory"],
        ])
        if concept_count >= 3:
            score += 1.5
        elif concept_count >= 2:
            score += 0.5

        # Classify
        if score >= 3.0:
            return "hard"
        elif score >= 1.5:
            return "medium"
        else:
            return "easy"


def classify_difficulty(problem_statement: str, classifier: Optional[DifficultyClassifier] = None) -> str:
    """
    Convenience function to classify problem difficulty.

    Args:
        problem_statement: Problem statement
        classifier: Optional trained classifier

    Returns:
        Difficulty label
    """
    if classifier is None:
        classifier = DifficultyClassifier()
    
    difficulty, _ = classifier.predict(problem_statement)
    return difficulty

