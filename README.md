# Reactor-BlindAI: Adversarial ML Security & Robustness in Nuclear SCADA

This project demonstrates **Adversarial Machine Learning (Evasion Attacks & Defense)** on an autonomous nuclear reactor cooling pump monitoring AI model. It illustrates how subtle, human-imperceptible perturbations can deceive neural networks, and how **Adversarial Training** can harden models against such threats.

## Architecture & Workflow

1. **Baseline AI Model (`NukePumpAI`):** A PyTorch Multi-Layer Perceptron (MLP) trained to classify reactor sensor telemetry into `NORMAL` (0) or `CRITICAL FAILURE` (1).
2. **Red Team (Attack - PGD/Iterative Evasion):** Using gradient ascent (`loss.backward()` on inputs), the model's perception is manipulated step-by-step to mistake a critical pump failure for a normal state.
3. **Blue Team (Defense - Adversarial Training):** The model is "vaccinated" by retraining it on the generated adversarial examples, establishing a robust decision boundary.

## Technologies Used
* **Python 3.12**
* **PyTorch** (Neural Network & Gradient Tracking)
* **NumPy**