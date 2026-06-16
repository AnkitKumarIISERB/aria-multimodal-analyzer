import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def add_realistic_noise(vector, noise_std=0.15):
    return np.clip(vector + np.random.normal(0, noise_std, vector.shape), -1.0, 1.0)

def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic training data for the conflict classifier with realistic noise.
    Features: [audio_score, face_score, abs_diff]
    Target: 1 if conflict, else 0
    """
    np.random.seed(42)
    
    # Base true emotions
    true_audio = np.random.uniform(-1.0, 1.0, num_samples)
    true_face = np.random.uniform(-1.0, 1.0, num_samples)
    
    # Ground truth label based on clean data
    y = (np.abs(true_audio - true_face) > 0.6).astype(int)
    
    # Add noise to simulate extraction inaccuracies
    noisy_audio = add_realistic_noise(true_audio, 0.25)
    noisy_face = add_realistic_noise(true_face, 0.25)
    
    # Also add some ambiguous boundary cases intentionally
    # 20 conflict, 20 non-conflict near the boundary
    bound_audio = np.random.uniform(-0.1, 0.1, 40)
    bound_face = np.array([0.55 if i < 20 else 0.65 for i in range(40)]) # Mix of just below and just above 0.6 diff
    bound_y = np.array([0 if i < 20 else 1 for i in range(40)])
    
    noisy_audio = np.concatenate([noisy_audio, bound_audio])
    noisy_face = np.concatenate([noisy_face, bound_face])
    y = np.concatenate([y, bound_y])
    
    abs_diff = np.abs(noisy_audio - noisy_face)
    X = np.column_stack((noisy_audio, noisy_face, abs_diff))
    
    return X, y

def train_and_save_model():
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(5000)
    
    print("Training Logistic Regression Classifier...")
    model = LogisticRegression(random_state=42)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(f"5-fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    assert cv_scores.mean() > 0.75, (
        f"Model CV F1 {cv_scores.mean():.3f} below threshold. "
        f"Check synthetic data generation."
    )
    
    model.fit(X, y)
    accuracy = model.score(X, y)
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    
    # Ensure directory exists
    os.makedirs('app/core', exist_ok=True)
    
    model_path = 'app/core/conflict_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
