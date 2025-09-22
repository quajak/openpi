import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.lax as lax
from typing import Optional, Tuple


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
        mask = jax.lax.stop_gradient(hard_mask - soft_mask) + soft_mask
        return mask
    else:
        return soft_mask


class AdaptiveTokenFilter(nn.Module):
    """Adaptive token filter with learnable k using Gumbel-Softmax top-k"""
    hidden_dim: int = 64
    
    def setup(self):
        self.scorer = nn.Sequential([
            nn.Dense(self.hidden_dim, dtype=jnp.float32),
            nn.relu,
            nn.Dense(1, dtype=jnp.float32),
        ])
    
    def __call__(self, token_embeddings: jnp.ndarray, tau: float = 1.0, training: bool = True,
                 rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
            token_embeddings: [batch_size, seq_len, embed_dim]
            tau: temperature for token selection
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

        expected_k = nn.sigmoid(importance_logits).sum(axis=-1)
        selection_mask = gumbel_softmax_topk(
            importance_logits, k=expected_k.astype(jnp.int32), tau=tau, hard=True, rng=rng2
        )
        
        # Apply mask to embeddings
        filtered_embeddings = token_embeddings * selection_mask[..., None]
        
        return filtered_embeddings, selection_mask, expected_k
