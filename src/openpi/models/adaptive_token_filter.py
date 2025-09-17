import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional

def gumbel_softmax_topk(logits: jnp.ndarray, k: int, tau: float = 1.0, 
                       hard: bool = False, rng: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Gumbel-Softmax Top-K sampling in JAX"""
    if rng is not None:
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape, minval=1e-8, maxval=1.0-1e-8)))
        perturbed_logits = logits + gumbel_noise
    else:
        perturbed_logits = logits
    
    soft_mask = jax.nn.softmax(perturbed_logits / tau, axis=-1)
    
    if hard:
        # Hard top-k with straight-through estimator
        top_indices = jnp.argsort(soft_mask, axis=-1)[..., -k:]
        hard_mask = jnp.zeros_like(soft_mask)
        hard_mask = hard_mask.at[jnp.arange(hard_mask.shape[0])[:, None], top_indices].set(1.0)
        
        # Straight-through: hard forward, soft backward
        mask = jax.lax.stop_gradient(hard_mask - soft_mask) + soft_mask
        return mask, top_indices
    else:
        return soft_mask, None

class AdaptiveTokenFilter(nn.Module):
    """Adaptive token filter with learnable k using Gumbel-Softmax top-k"""
    hidden_dim: int = 64
    max_k: int = 10  # Maximum number of tokens to select
    
    def setup(self):
        self.scorer = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(1)
        ])
        # Learnable parameter for k (logits over possible k values)
        self.k_logits = self.param('k_logits', nn.initializers.zeros, (self.max_k,))
    
    def __call__(self, 
                 token_embeddings: jnp.ndarray,
                 tau: float = 1.0,
                 k_tau: float = 1.0,
                 training: bool = True,
                 rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
            token_embeddings: [batch_size, seq_len, embed_dim]
            tau: temperature for token selection
            k_tau: temperature for k selection
            training: whether in training mode
            rng: random key for Gumbel noise
            
        Returns:
            filtered_embeddings: [batch_size, seq_len, embed_dim]
            selection_mask: [batch_size, seq_len] 
            expected_k: scalar - expected number of tokens kept
        """
        batch_size, seq_len, embed_dim = token_embeddings.shape
        
        # Score each token's importance
        importance_logits = self.scorer(token_embeddings).squeeze(-1)  # [batch, seq_len]
        
        if training and rng is not None:
            rng1, rng2 = jax.random.split(rng)
        else:
            rng1, rng2 = None, None
        
        # Sample k using Gumbel-Softmax over possible k values
        if training and rng1 is not None:
            k_probs = jax.nn.softmax(self.k_logits / k_tau)
            k_gumbel = -jnp.log(-jnp.log(jax.random.uniform(rng1, self.k_logits.shape, minval=1e-8, maxval=1.0-1e-8)))
            k_soft = jax.nn.softmax((self.k_logits + k_gumbel) / k_tau)
            # Expected k (differentiable)
            expected_k = jnp.sum(k_soft * jnp.arange(1, self.max_k + 1))
            # For actual selection, use argmax (hard k)
            k_selected = jnp.argmax(k_soft) + 1
        else:
            k_probs = jax.nn.softmax(self.k_logits)
            expected_k = jnp.sum(k_probs * jnp.arange(1, self.max_k + 1))
            k_selected = jnp.argmax(k_probs) + 1
        
        # Apply Gumbel-Softmax top-k with the selected k
        selection_mask, _ = gumbel_softmax_topk(
            importance_logits, k=k_selected, tau=tau, hard=True, rng=rng2
        )
        
        # Apply mask to embeddings
        filtered_embeddings = token_embeddings * selection_mask[..., None]
        
        return filtered_embeddings, selection_mask, expected_k