"""
Toy task to test the adaptive token filter with a small transformer.

This test creates a classification task where:
- Input sequences contain both useful signal tokens and useless noise tokens
- The adaptive token filter should learn to select the useful tokens
- A small transformer processes the filtered tokens for classification
"""

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from openpi.models.adaptive_token_filter import AdaptiveTokenFilter


def setup_gpu():
    """Configure JAX to use GPU with optimal memory settings."""
    # Set XLA memory fraction to use more GPU memory (90% instead of default 75%)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    
    # Check if GPU is available
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'NVIDIA' in d.device_kind]
        if gpu_devices:
            print(f"Using GPU: {gpu_devices[0]}")
            return gpu_devices[0]
        else:
            print("No GPU found, using CPU")
            return jax.devices('cpu')[0]
    except Exception as e:
        print(f"Error detecting GPU: {e}, falling back to CPU")
        return jax.devices('cpu')[0]


# Set up GPU configuration
device = setup_gpu()


class ToyTransformer(nn.Module):
    """Small transformer for testing adaptive token filter."""
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    num_classes: int
    max_seq_len: int
    use_adaptive_filter: bool = True
    filter_hidden_dim: int = 64
    filter_max_k: int = 10
    
    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_dim)
        self.pos_embedding = nn.Embed(self.max_seq_len, self.embed_dim)
        
        if self.use_adaptive_filter:
            self.token_filter = AdaptiveTokenFilter(
                hidden_dim=self.filter_hidden_dim,
                max_k=self.filter_max_k
            )
        
        # Transformer layers
        self.transformer_layers = [
            nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                out_features=self.embed_dim
            )
            for _ in range(self.num_layers)
        ]
        self.layer_norms = [nn.LayerNorm() for _ in range(self.num_layers)]
        self.mlp_layers = [
            nn.Sequential([
                nn.Dense(self.embed_dim * 4),
                nn.relu,
                nn.Dense(self.embed_dim)
            ]) for _ in range(self.num_layers)
        ]
        
        # Classification head
        self.classifier = nn.Dense(self.num_classes)
    
    def __call__(self, tokens: jnp.ndarray, mask: jnp.ndarray, 
                 training: bool = True, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Args:
            tokens: [batch_size, seq_len] token indices
            mask: [batch_size, seq_len] attention mask
            training: whether in training mode
            rng: random key for token filtering
            
        Returns:
            logits: [batch_size, num_classes] classification logits
            metrics: dict with filter statistics
        """
        assert rng is not None
        batch_size, seq_len = tokens.shape
        
        # Embed tokens
        x = self.embedding(tokens)  # [batch, seq_len, embed_dim]
        
        # Add positional embeddings
        positions = jnp.arange(seq_len)[None, :]
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        # Apply adaptive token filter if enabled
        if self.use_adaptive_filter:
            rng1, rng2 = jax.random.split(rng)
            filtered_x, selection_mask, expected_k = self.token_filter(
                x, tau=1.0, training=training, rng=rng2
            )
            x = filtered_x
            
            # Update mask to include filter selection
            # mask = mask & selection_mask.astype(jnp.bool_)
            
            # Calculate metrics directly (no state storage)
            kept_fraction = jnp.mean(selection_mask)
        else:
            # No filtering - all tokens kept
            expected_k = jnp.array(0.0)
            kept_fraction = jnp.array(1.0)
        
        # Apply transformer layers
        for i, (attn, layer_norm, mlp) in enumerate(zip(self.transformer_layers, self.layer_norms, self.mlp_layers)):
            # Self-attention
            attn_out = attn(x, mask=jnp.reshape(jnp.tile(mask, mask.shape[-1]*self.num_heads), (mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1])))
            x = layer_norm(x + attn_out)
            
            # MLP
            mlp_out = mlp(x)
            x = layer_norm(x + mlp_out)
        
        # Global average pooling (masked)
        mask_expanded = mask[..., None]
        pooled = jnp.sum(x * mask_expanded, axis=1) / jnp.sum(mask_expanded, axis=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        metrics = {
            'expected_k': expected_k,
            'kept_fraction': kept_fraction,
            'active_tokens': jnp.sum(mask, axis=1)
        }
        
        return logits, expected_k, metrics


def generate_toy_data(batch_size: int, seq_len: int, vocab_size: int, 
                     num_classes: int, signal_strength: float = 2.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic data with useful signal tokens and noise tokens.
    
    Args:
        batch_size: number of samples
        seq_len: sequence length
        vocab_size: vocabulary size
        num_classes: number of classes
        signal_strength: how strong the signal tokens are
        
    Returns:
        tokens: [batch_size, seq_len] token indices
        labels: [batch_size] class labels
        signal_mask: [batch_size, seq_len] which tokens are signal vs noise
    """
    rng = jax.random.PRNGKey(42)
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    
    # Generate random tokens
    tokens = jax.random.randint(rng1, (batch_size, seq_len), 0, vocab_size)
    
    # Generate class labels
    labels = jax.random.randint(rng2, (batch_size,), 0, num_classes)
    
    # Create signal tokens based on class labels
    # Each class has specific "signal" token patterns
    signal_tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
    
    # First 5 tokens are always signal tokens (class-specific)
    for i in range(5):
        # Each class gets different signal tokens
        class_signal_tokens = labels * 10 + i  # Different token range per class
        signal_tokens = signal_tokens.at[:, i].set(True)
        tokens = tokens.at[:, i].set(class_signal_tokens)
    
    # Add some random signal tokens in the middle
    middle_signal_pos = jax.random.randint(rng3, (batch_size, 3), 5, seq_len-5)
    for i in range(3):
        batch_indices = jnp.arange(batch_size)
        signal_tokens = signal_tokens.at[batch_indices, middle_signal_pos[:, i]].set(True)
        # Make these tokens class-specific too
        class_signal_tokens = labels * 10 + 5 + i
        tokens = tokens.at[batch_indices, middle_signal_pos[:, i]].set(class_signal_tokens)
    
    # Ensure arrays are on the correct device
    tokens = jax.device_put(tokens, device)
    labels = jax.device_put(labels, device)
    signal_tokens = jax.device_put(signal_tokens, device)
    
    return tokens, labels, signal_tokens


def create_model(config: Dict[str, Any], rng: jax.random.PRNGKey) -> Tuple[ToyTransformer, Dict[str, Any]]:
    """Create and initialize the model."""
    model = ToyTransformer(**config)
    
    # Initialize with dummy data on the correct device
    dummy_tokens = jax.device_put(jnp.zeros((1, config['max_seq_len']), dtype=jnp.int32), device)
    dummy_mask = jax.device_put(jnp.ones((1, config['max_seq_len']), dtype=jnp.bool_), device)
    
    params = model.init(rng, dummy_tokens, dummy_mask, training=True, rng=rng)
    
    return model, params


def train_step(model: ToyTransformer, params: Dict[str, Any], 
               tokens: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray,
               optimizer: optax.GradientTransformation, opt_state: Any,
               rng: jax.random.PRNGKey) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Single training step."""
    
    def loss_fn(params):
        logits, expected_k, metrics = model.apply(params, tokens, mask, training=True, rng=rng)
        task_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        metrics['task_loss'] = task_loss
        metrics['expected_k_loss'] = 0.0001 * expected_k.mean()
        loss = task_loss + 0.0001 * expected_k.mean()
        return loss, (logits, metrics)
    
    (loss, (logits, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # Compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    
    metrics.update({
        'loss': loss,
        'accuracy': accuracy
    })
    
    return params, opt_state, metrics


# JIT compile the training step for better GPU performance
train_step_jit = jax.jit(train_step, static_argnums=(0, 5))


def evaluate(model: ToyTransformer, params: Dict[str, Any],
             tokens: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray, rng: jax.random.PRNGKey) -> Dict[str, Any]:
    """Evaluate the model."""
    logits, expected_k, metrics = model.apply(params, tokens, mask, training=False, rng=rng)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == labels)
    
    metrics.update({
        'loss': loss,
        'accuracy': accuracy,
        'expected_k': expected_k.mean()
    })
    
    return metrics


# JIT compile the evaluation function for better GPU performance
evaluate_jit = jax.jit(evaluate, static_argnums=(0,))


def visualize_token_selection(tokens: jnp.ndarray, signal_mask: jnp.ndarray, 
                            selection_mask: jnp.ndarray, labels: jnp.ndarray,
                            predictions: jnp.ndarray, sample_idx: int = 0):
    """Visualize which tokens were selected vs which were signal."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    seq_len = tokens.shape[1]
    x = jnp.arange(seq_len)
    
    # Plot 1: Signal tokens vs selected tokens
    ax1.scatter(x[signal_mask[sample_idx]], jnp.ones(jnp.sum(signal_mask[sample_idx])), 
               c='green', label='Signal tokens', s=50, alpha=0.7)
    ax1.scatter(x[selection_mask[sample_idx]], jnp.zeros(jnp.sum(selection_mask[sample_idx])), 
               c='blue', label='Selected tokens', s=50, alpha=0.7)
    ax1.set_ylabel('Token Type')
    ax1.set_xlabel('Position')
    ax1.set_title(f'Sample {sample_idx}: True label={labels[sample_idx]}, Pred={predictions[sample_idx]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Token values
    colors = ['red' if not signal_mask[sample_idx, i] else 'green' for i in range(seq_len)]
    ax2.bar(x, tokens[sample_idx], color=colors, alpha=0.7)
    ax2.set_ylabel('Token Value')
    ax2.set_xlabel('Position')
    ax2.set_title('Token Values (Red=Noise, Green=Signal)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'token_selection.png')


def main():
    """Main training and evaluation loop."""
    
    # Configuration
    config = {
        'vocab_size': 100,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'num_classes': 5,
        'max_seq_len': 20,
        'use_adaptive_filter': True,
        'filter_hidden_dim': 32,
        'filter_max_k': 8
    }
    
    # Training parameters
    batch_size = 32
    num_epochs = 200
    learning_rate = 1e-3
    
    # Create model
    rng = jax.random.PRNGKey(0)
    model, params = create_model(config, rng)
    
    # Generate data
    print("Generating synthetic data...")
    train_tokens, train_labels, train_signal_mask = generate_toy_data(
        batch_size * 10, config['max_seq_len'], config['vocab_size'], 
        config['num_classes'], signal_strength=1.1
    )
    val_tokens, val_labels, val_signal_mask = generate_toy_data(
        batch_size * 2, config['max_seq_len'], config['vocab_size'], 
        config['num_classes'], signal_strength=1.1
    )
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Training loop
    print("Starting training...")
    train_mask = jax.device_put(jnp.ones_like(train_tokens, dtype=jnp.bool_), device)
    val_mask = jax.device_put(jnp.ones_like(val_tokens, dtype=jnp.bool_), device)
    
    for epoch in range(num_epochs):
        # Shuffle training data
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, len(train_tokens))
        train_tokens = train_tokens[perm]
        train_labels = train_labels[perm]
        train_signal_mask = train_signal_mask[perm]
        
        # Training
        epoch_metrics = []
        for i in range(0, len(train_tokens), batch_size):
            batch_tokens = train_tokens[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            batch_mask = train_mask[i:i+batch_size]
            
            step_rng, rng = jax.random.split(rng)
            params, opt_state, metrics = train_step_jit(
                model, params, batch_tokens, batch_labels, batch_mask, 
                optimizer, opt_state, step_rng
            )
            epoch_metrics.append(metrics)
        
        # Average metrics for epoch
        avg_metrics = jax.tree.map(lambda *xs: jnp.mean(jnp.array(xs)), *epoch_metrics)
        
        # Validation
        eval_rng, rng = jax.random.split(rng)
        val_metrics = evaluate_jit(model, params, val_tokens, val_labels, val_mask, eval_rng)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: "
                  f"Train Loss={avg_metrics['loss']:.4f}, "
                  f"Train Task Loss={avg_metrics['task_loss']:.4f}, "
                  f"Train Expected K Loss={avg_metrics['expected_k_loss']:.4f}, "
                  f"Train Acc={avg_metrics['accuracy']:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}, "
                  f"Expected K={avg_metrics['expected_k']:.2f}, "
                  f"Kept Frac={avg_metrics['kept_fraction']:.3f}")
    
    # Final evaluation with token selection visualization
    print("\nFinal evaluation...")
    eval_rng, rng = jax.random.split(rng)
    final_metrics = evaluate_jit(model, params, val_tokens, val_labels, val_mask, eval_rng)
    print(f"Final validation accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final expected K: {final_metrics['expected_k'].item():.2f}")
    print(f"Final kept fraction: {final_metrics['kept_fraction']:.3f}")
    
    # Get token selection for visualization
    eval_rng, rng = jax.random.split(rng)
    logits, expected_k, metrics = model.apply(params, val_tokens[:1], val_mask[:1], training=False, rng=eval_rng)
    predictions = jnp.argmax(logits, axis=-1)
    
    # Note: In evaluation mode, we don't get selection masks, so we'll use signal masks for demo
    print("\nVisualizing token selection (using signal masks as proxy)...")
    visualize_token_selection(val_tokens[:1], val_signal_mask[:1], 
                            val_signal_mask[:1], val_labels[:1], predictions[:1])


if __name__ == "__main__":
    main()
