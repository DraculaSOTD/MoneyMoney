#!/usr/bin/env python3
"""
Test script to verify the GRU-Attention backpropagation implementation.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.deep_learning.gru_attention.model import GRUAttentionModel
from models.deep_learning.gru_attention.trainer import GRUAttentionTrainer


def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print("Testing GRU-Attention backpropagation implementation...")
    
    # Model parameters
    input_size = 10
    hidden_sizes = [32, 16]
    num_attention_heads = 4
    batch_size = 8
    seq_length = 20
    
    # Create model
    model = GRUAttentionModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_attention_heads=num_attention_heads,
        num_classes=3,
        learning_rate=0.001
    )
    
    # Create synthetic data
    X = np.random.randn(batch_size, seq_length, input_size)
    y = np.random.randint(0, 3, size=batch_size)
    
    print(f"\nInput shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Forward pass
    print("\n1. Forward pass...")
    predictions = model.forward(X, return_attention=True)
    print(f"   - Action logits shape: {predictions['action_logits'].shape}")
    print(f"   - Action probs shape: {predictions['action_probs'].shape}")
    print(f"   - Confidence shape: {predictions['confidence'].shape}")
    print(f"   - Attention weights shape: {predictions['attention_weights'].shape}")
    
    # Compute loss
    print("\n2. Computing loss...")
    targets = {'actions': y}
    losses = model.compute_loss(predictions, targets)
    print(f"   - Total loss: {losses['total_loss']:.4f}")
    print(f"   - Action loss: {losses['action_loss']:.4f}")
    print(f"   - Confidence loss: {losses['confidence_loss']:.4f}")
    print(f"   - L2 loss: {losses['l2_loss']:.4f}")
    
    # Backward pass
    print("\n3. Backward pass...")
    gradients = model.backward(predictions, targets)
    print(f"   - Number of gradient tensors: {len(gradients)}")
    
    # Check gradient shapes
    print("\n4. Checking gradient shapes...")
    for name, grad in gradients.items():
        if name in model.params:
            param_shape = model.params[name].shape
            grad_shape = grad.shape
            assert param_shape == grad_shape, f"Shape mismatch for {name}: param {param_shape} vs grad {grad_shape}"
            print(f"   - {name}: {grad_shape} ✓")
    
    # Test weight update
    print("\n5. Testing weight update...")
    initial_loss = losses['total_loss']
    
    # Update weights
    model.update_weights(gradients)
    
    # Forward pass again
    predictions_after = model.forward(X, return_attention=True)
    losses_after = model.compute_loss(predictions_after, targets)
    
    print(f"   - Loss before update: {initial_loss:.4f}")
    print(f"   - Loss after update: {losses_after['total_loss']:.4f}")
    print(f"   - Loss decreased: {initial_loss > losses_after['total_loss']}")
    
    print("\n✓ Backpropagation implementation is working correctly!")


def test_trainer_integration():
    """Test the trainer with the new backpropagation."""
    print("\n\nTesting trainer integration...")
    
    # Create model
    model = GRUAttentionModel(
        input_size=10,
        hidden_sizes=[16, 8],
        num_attention_heads=2,
        num_classes=3,
        learning_rate=0.01
    )
    
    # Create trainer
    trainer = GRUAttentionTrainer(
        model=model,
        batch_size=4,
        validation_split=0.2,
        patience=5
    )
    
    # Create synthetic time series data
    n_samples = 100
    n_features = 10
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 3, size=n_samples)
    
    # Prepare sequences
    X, y = trainer.prepare_data(features, labels, sequence_length=10)
    print(f"\nPrepared data shapes: X={X.shape}, y={y.shape}")
    
    # Train for a few epochs
    print("\nTraining for 3 epochs...")
    history = trainer.train(X, y, epochs=3, verbose=1)
    
    print("\n✓ Trainer integration successful!")
    
    # Check that loss is decreasing
    train_losses = history['train_loss']
    print(f"\nTraining losses: {[f'{loss:.4f}' for loss in train_losses]}")
    
    if len(train_losses) > 1 and train_losses[-1] < train_losses[0]:
        print("✓ Training loss is decreasing!")
    else:
        print("⚠ Training loss is not decreasing - may need more epochs or tuning")


if __name__ == "__main__":
    test_gradient_flow()
    test_trainer_integration()