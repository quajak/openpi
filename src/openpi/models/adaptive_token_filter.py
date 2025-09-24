import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.lax as lax
from typing import Optional, Tuple
import flax.nnx as nnx


class MLP(nnx.Module):
    """Multi-layer perceptron using nnx.Linear layers"""
    
    def __init__(self, features: list[int], *, rngs: nnx.Rngs):
        super().__init__()
        self.features = features
        self.layers = []
        
        for i in range(len(features) - 1):
            self.layers.append(nnx.Linear(features[i], features[i + 1], rngs=rngs))
    
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward pass through MLP layers"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU activation except for the last layer
            if i < len(self.layers) - 1:
                x = nnx.relu(x)
        return x


def gumbel_softmax_topk(logits: jnp.ndarray, k: jnp.ndarray, tau: float = 1.0, 
                       hard: bool = False, rng: jax.random.PRNGKey = None) -> jnp.ndarray:
    """Gumbel-Softmax Top-K sampling in JAX"""
    if rng is not None:
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape, minval=1e-8, maxval=1.0-1e-8)))
        perturbed_logits = logits + gumbel_noise
    else:
        perturbed_logits = logits

    soft_mask = jax.nn.softmax(perturbed_logits / tau, axis=-1)
    
    if hard:
        sorted = jnp.argsort(soft_mask, axis=-1, descending=True)
        def set_and_increment(acc, i, ind):
            acc = acc.at[i, sorted[i, ind]].set(1.0)
            return acc, ind + 1
        def one_line_topk(acc, i):
            return lax.while_loop(lambda ind: ind[1] < k[i], lambda ind: set_and_increment(ind[0], i, ind[1]), (acc, 0))[0]
        hard_mask = lax.fori_loop(0, soft_mask.shape[0], lambda i, acc: one_line_topk(acc, i), jnp.zeros_like(soft_mask))
        
        # Straight-through: hard forward, soft backward
        mask = lax.stop_gradient(hard_mask - soft_mask) + soft_mask
        return mask
    else:
        return soft_mask


class TokenCountValueFunction(nnx.Module):
    """Transformer on image tokens predicting residual loss increases when tokens are removed"""
    max_tokens: int = 256
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    
    def __init__(self, max_tokens: int = 256, hidden_dim: int = 64, num_layers: int = 2, num_heads: int = 4, *, rngs: nnx.Rngs):
        super().__init__()
        self.max_tokens = max_tokens
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Transformer layers with batch normalization
        self.transformer_layers = []
        self.batch_norms = []
        for _ in range(num_layers):
            self.transformer_layers.append(nnx.MultiHeadAttention(
                num_heads=num_heads,
                in_features=hidden_dim,
                rngs=rngs
            ))
            self.batch_norms.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
            self.transformer_layers.append(MLP([hidden_dim * 4, hidden_dim], rngs=rngs))
            self.batch_norms.append(nnx.BatchNorm(hidden_dim, rngs=rngs))
        
        # MLP to predict residual loss increases for each token position
        self.residual_mlp = MLP([hidden_dim, hidden_dim * 2, max_tokens], rngs=rngs)
    
    def __call__(self, image_tokens: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            image_tokens: [batch_size, seq_len, embed_dim] - image tokens
            
        Returns:
            predicted_losses: [batch_size, max_tokens] - predicted cumulative loss for each token count
        """
        batch_size, seq_len, embed_dim = image_tokens.shape
        
        # Apply transformer layers with batch normalization
        x = image_tokens
        bn_idx = 0
        for i in range(0, len(self.transformer_layers), 2):
            attn_layer = self.transformer_layers[i]
            mlp_layer = self.transformer_layers[i + 1]
            
            # Self-attention
            x = attn_layer(x, training=training)
            x = self.batch_norms[bn_idx](x, training=training)
            x = nnx.relu(x)
            bn_idx += 1
            
            # MLP
            x = mlp_layer(x, training=training)
            x = self.batch_norms[bn_idx](x, training=training)
            x = nnx.relu(x)
            bn_idx += 1
        
        # Elementwise max over all tokens
        x_max = jnp.max(x, axis=1)  # [batch_size, hidden_dim]
        
        # Predict residual loss increases for each token position
        residual_losses = self.residual_mlp(x_max).clip(0)  # [batch_size, max_tokens]
        
        # Apply cumulative sum to get predicted losses for each token count
        predicted_losses = jnp.cumsum(residual_losses, axis=-1)  # [batch_size, max_tokens]
        
        return predicted_losses


class AdaptiveTokenFilter(nn.Module):
    """Adaptive token filter with learnable k using Gumbel-Softmax top-k and value function"""
    hidden_dim: int = 64
    max_tokens: int = 256
    use_value_function: bool = True
    random_k_prob: float = 1.0  # Probability of using random k selection
    random_k_decay: float = 0.99  # Decay rate for random k probability
    
    def setup(self):
        self.scorer = nn.Sequential([
            nn.Dense(self.hidden_dim, dtype=jnp.float32),
            nn.relu,
            nn.Dense(1, dtype=jnp.float32),
        ])
        
        if self.use_value_function:
            # Initialize value function (will be set externally)
            self.value_function = None
    
    def set_value_function(self, value_function: TokenCountValueFunction):
        """Set the value function model"""
        self.value_function = value_function
    
    def set_random_k_prob(self, prob: float):
        """Update the probability of using random k selection"""
        self.random_k_prob = prob
    
    def __call__(self, token_embeddings: jnp.ndarray, tau: float = 1.0, training: bool = True,
                 rng: Optional[jax.random.PRNGKey] = None, step: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """
        Args:
            token_embeddings: [batch_size, seq_len, embed_dim]
            tau: temperature for token selection
            training: whether in training mode
            rng: random key for Gumbel noise
            step: training step for random k probability decay
            
        Returns:
            filtered_embeddings: [batch_size, seq_len, embed_dim]
            selection_mask: [batch_size, seq_len] 
            expected_k: [batch_size] - expected number of tokens kept per image
            info: dict with additional metrics including predicted_losses
        """
        batch_size, seq_len, embed_dim = token_embeddings.shape
        
        # Score each token's importance
        importance_logits = self.scorer(token_embeddings).squeeze(-1)  # [batch, seq_len]
        
        if training and rng is not None:
            rng, rng2, rng3 = jax.random.split(rng, 3)
        else:
            rng, rng2, rng3 = None, None, None

        # Calculate expected_k based on value function or random selection
        predicted_losses = None
        if self.use_value_function and self.value_function is not None and training:
            # Use value function to predict optimal k for each image
            current_random_prob = self.random_k_prob * (self.random_k_decay ** step)
            use_random = jax.random.bernoulli(rng3, current_random_prob)
            
            if use_random:
                # Random k selection for exploration
                expected_k = jax.random.uniform(rng3, (batch_size,), minval=1, maxval=seq_len)
            else:
                # Get predicted losses for all token counts
                predicted_losses = self.value_function(token_embeddings, training=training)  # [batch_size, max_tokens]
                
                # Normalize losses between 0 and 1 for each image
                min_losses = jnp.min(predicted_losses, axis=-1, keepdims=True)  # [batch_size, 1]
                max_losses = jnp.max(predicted_losses, axis=-1, keepdims=True)  # [batch_size, 1]
                normalized_losses = (predicted_losses - min_losses) / (max_losses - min_losses + 1e-8)
                
                # Find first k where normalized loss <= 0.2 for each image
                loss_mask = normalized_losses <= 0.2
                expected_k = jnp.argmax(loss_mask.astype(jnp.int32), axis=-1) + 1
                expected_k = jnp.clip(expected_k, 1, seq_len)
        else:
            # Fallback to original method
            expected_k = nn.sigmoid(importance_logits).sum(axis=-1)
        
        selection_mask = gumbel_softmax_topk(
            importance_logits, k=jnp.clip(expected_k.astype(jnp.int32), min=32), tau=tau, hard=True, rng=rng2
        )
        
        # Apply mask to embeddings
        filtered_embeddings = token_embeddings * selection_mask[..., None]
        
        # Prepare info dict
        info = {
            "expected_k": expected_k.mean(),
            "kept_fraction": selection_mask.mean(),
            "selection_mask": selection_mask,  # Pass selection mask for loss calculation
        }
        
        if self.use_value_function and self.value_function is not None and training:
            current_random_prob = self.random_k_prob * (self.random_k_decay ** step)
            info["random_k_prob"] = current_random_prob
            info["use_random_k"] = use_random if rng3 is not None else False
            
            # Pass predicted losses for loss calculation
            if predicted_losses is not None:
                info["predicted_losses"] = predicted_losses
            
            # Log distribution occasionally
            if step % 100 == 0 and predicted_losses is not None:
                # Sample one image for distribution logging
                sample_idx = jax.random.randint(rng3, (), 0, batch_size)
                sample_predicted = predicted_losses[sample_idx]  # [max_tokens]
                
                # Normalize for visualization
                min_loss = jnp.min(sample_predicted)
                max_loss = jnp.max(sample_predicted)
                normalized_losses = (sample_predicted - min_loss) / (max_loss - min_loss + 1e-8)
                
                token_counts = jnp.arange(1, min(seq_len + 1, self.max_tokens + 1))
                
                info["value_function_distribution"] = {
                    "token_counts": token_counts,
                    "predicted_losses": sample_predicted[:len(token_counts)],
                    "normalized_losses": normalized_losses[:len(token_counts)]
                }
        
        return filtered_embeddings, selection_mask, expected_k, info
