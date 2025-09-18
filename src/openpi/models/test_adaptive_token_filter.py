import pytest
import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Tuple
from openpi.models.adaptive_token_filter import gumbel_softmax_topk, AdaptiveTokenFilter


class TestGumbelSoftmaxTopK:
    """Test cases for gumbel_softmax_topk function"""
    
    def test_basic_functionality(self):
        """Test basic functionality with simple inputs"""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        k = jnp.array([2])
        rng = jax.random.PRNGKey(42)
        
        result = gumbel_softmax_topk(logits, k, tau=1.0, hard=False, rng=rng)
        
        # Check shape
        assert result.shape == logits.shape
        # Check that values sum to approximately 1 (softmax property)
        assert jnp.allclose(result.sum(axis=-1), 1.0, atol=1e-6)
        # Check that values are non-negative
        assert jnp.all(result >= 0)
    
    def test_hard_mode(self):
        """Test hard mode returns binary mask"""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        k = jnp.array([2])
        rng = jax.random.PRNGKey(42)
        
        result = gumbel_softmax_topk(logits, k, tau=1.0, hard=True, rng=rng)
        
        # Check shape
        assert result.shape == logits.shape
        # Check that values are binary (0 or 1)
        assert jnp.all((result == 0) | (result == 1))
        # Check that exactly k values are 1
        assert jnp.all(result.sum(axis=-1) == k)
    
    def test_batch_processing(self):
        """Test processing multiple samples in batch"""
        logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]])
        k = jnp.array([2, 1])
        rng = jax.random.PRNGKey(42)
        
        result = gumbel_softmax_topk(logits, k, tau=1.0, hard=True, rng=rng)
        
        # Check shape
        assert result.shape == logits.shape
        # Check that each row has correct number of selected tokens
        assert jnp.all(result.sum(axis=-1) == k)
    
    def test_different_k_values(self):
        """Test with different k values including edge cases"""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        # Test k=0 (should select nothing)
        k_zero = jnp.array([0])
        rng = jax.random.PRNGKey(42)
        result_zero = gumbel_softmax_topk(logits, k_zero, tau=1.0, hard=True, rng=rng)
        assert jnp.all(result_zero == 0)
        
        # Test k=seq_len (should select all)
        k_all = jnp.array([logits.shape[-1]])
        result_all = gumbel_softmax_topk(logits, k_all, tau=1.0, hard=True, rng=rng)
        assert jnp.all(result_all == 1)
    
    def test_temperature_effects(self):
        """Test that temperature affects the distribution"""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        k = jnp.array([2])
        rng = jax.random.PRNGKey(42)
        
        # Low temperature should be more deterministic
        result_low_tau = gumbel_softmax_topk(logits, k, tau=0.1, hard=False, rng=rng)
        # High temperature should be more uniform
        result_high_tau = gumbel_softmax_topk(logits, k, tau=10.0, hard=False, rng=rng)
        
        # Both should still sum to 1
        assert jnp.allclose(result_low_tau.sum(axis=-1), 1.0)
        assert jnp.allclose(result_high_tau.sum(axis=-1), 1.0)
    
    def test_no_rng_provided(self):
        """Test behavior when no random key is provided"""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        k = jnp.array([2])
        
        result = gumbel_softmax_topk(logits, k, tau=1.0, hard=False, rng=None)
        
        # Should still work and return softmax
        assert result.shape == logits.shape
        assert jnp.allclose(result.sum(axis=-1), 1.0)
    
    def test_deterministic_with_same_rng(self):
        """Test that same RNG produces same results"""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        k = jnp.array([2])
        rng = jax.random.PRNGKey(42)
        
        result1 = gumbel_softmax_topk(logits, k, tau=1.0, hard=True, rng=rng)
        result2 = gumbel_softmax_topk(logits, k, tau=1.0, hard=True, rng=rng)
        
        # Results should be identical with same RNG
        assert jnp.allclose(result1, result2)


class TestAdaptiveTokenFilter:
    """Test cases for AdaptiveTokenFilter class"""
    
    def test_initialization(self):
        """Test proper initialization of the filter"""
        filter_model = AdaptiveTokenFilter(hidden_dim=32, max_k=5)
        
        # Check that attributes are set correctly
        assert filter_model.hidden_dim == 32
        assert filter_model.max_k == 5
    
    def test_forward_pass_shape(self):
        """Test forward pass returns correct shapes"""
        batch_size, seq_len, embed_dim = 2, 10, 64
        token_embeddings = jnp.ones((batch_size, seq_len, embed_dim))
        rng = jax.random.PRNGKey(42)
        
        filter_model = AdaptiveTokenFilter(hidden_dim=32, max_k=5)
        params = filter_model.init(rng, token_embeddings)
        
        filtered_embeddings, selection_mask, expected_k = filter_model.apply(
            params, token_embeddings, rng=rng
        )
        
        # Check output shapes
        assert filtered_embeddings.shape == token_embeddings.shape
        assert selection_mask.shape == (batch_size, seq_len)
        assert expected_k.shape == (batch_size,)
    
    def test_selection_mask_properties(self):
        """Test that selection mask has correct properties"""
        batch_size, seq_len, embed_dim = 1, 8, 32
        token_embeddings = jnp.ones((batch_size, seq_len, embed_dim))
        rng = jax.random.PRNGKey(42)
        
        filter_model = AdaptiveTokenFilter(hidden_dim=16, max_k=3)
        params = filter_model.init(rng, token_embeddings)
        
        filtered_embeddings, selection_mask, expected_k = filter_model.apply(
            params, token_embeddings, rng=rng
        )
        
        # Selection mask should be binary
        assert jnp.all((selection_mask == 0) | (selection_mask == 1))
        # Number of selected tokens should match expected_k
        assert jnp.allclose(selection_mask.sum(axis=-1), int(expected_k.item()), atol=1e-6)
    
    def test_filtering_effect(self):
        """Test that filtering actually affects the embeddings"""
        batch_size, seq_len, embed_dim = 1, 5, 8
        token_embeddings = jnp.ones((batch_size, seq_len, embed_dim))
        rng = jax.random.PRNGKey(42)
        
        filter_model = AdaptiveTokenFilter(hidden_dim=16, max_k=2)
        params = filter_model.init(rng, token_embeddings)
        
        filtered_embeddings, selection_mask, expected_k = filter_model.apply(
            params, token_embeddings, rng=rng
        )
        
        # Filtered embeddings should be different from original
        assert not jnp.allclose(filtered_embeddings, token_embeddings)
        # Filtered embeddings should be zero where mask is zero
        assert jnp.allclose(filtered_embeddings * (1 - selection_mask[..., None]), 0)
    
    def test_training_vs_inference(self):
        """Test different behavior in training vs inference mode"""
        batch_size, seq_len, embed_dim = 1, 6, 16
        token_embeddings = jnp.ones((batch_size, seq_len, embed_dim))
        rng = jax.random.PRNGKey(42)
        
        filter_model = AdaptiveTokenFilter(hidden_dim=16, max_k=3)
        params = filter_model.init(rng, token_embeddings)
        
        # Training mode
        _, mask_train, _ = filter_model.apply(
            params, token_embeddings, training=True, rng=rng
        )
        
        # Inference mode (no RNG)
        _, mask_inference, _ = filter_model.apply(
            params, token_embeddings, training=False, rng=None
        )
        
        # Both should work without errors
        assert mask_train.shape == (batch_size, seq_len)
        assert mask_inference.shape == (batch_size, seq_len)
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths"""
        embed_dim = 32
        rng = jax.random.PRNGKey(42)
        
        filter_model = AdaptiveTokenFilter(hidden_dim=16, max_k=2)
        
        # Test with short sequence
        short_embeddings = jnp.ones((1, 3, embed_dim))
        params = filter_model.init(rng, short_embeddings)
        _, mask_short, _ = filter_model.apply(params, short_embeddings, rng=rng)
        assert mask_short.shape == (1, 3)
        
        # Test with longer sequence
        long_embeddings = jnp.ones((1, 20, embed_dim))
        _, mask_long, _ = filter_model.apply(params, long_embeddings, rng=rng)
        assert mask_long.shape == (1, 20)
    
    def test_temperature_parameters(self):
        """Test different temperature parameters"""
        batch_size, seq_len, embed_dim = 1, 8, 16
        token_embeddings = jnp.ones((batch_size, seq_len, embed_dim))
        rng = jax.random.PRNGKey(42)
        
        filter_model = AdaptiveTokenFilter(hidden_dim=16, max_k=3)
        params = filter_model.init(rng, token_embeddings)
        
        # Test with different tau values
        _, mask_low_tau, _ = filter_model.apply(
            params, token_embeddings, tau=0.1, rng=rng
        )
        _, mask_high_tau, _ = filter_model.apply(
            params, token_embeddings, tau=10.0, rng=rng
        )
        
        # Both should work
        assert mask_low_tau.shape == (batch_size, seq_len)
        assert mask_high_tau.shape == (batch_size, seq_len)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the filter"""
        batch_size, seq_len, embed_dim = 1, 5, 8
        token_embeddings = jnp.ones((batch_size, seq_len, embed_dim))
        rng = jax.random.PRNGKey(42)
        
        filter_model = AdaptiveTokenFilter(hidden_dim=16, max_k=2)
        params = filter_model.init(rng, token_embeddings)
        
        def loss_fn(params):
            filtered_embeddings, _, _ = filter_model.apply(
                params, token_embeddings, rng=rng
            )
            return jnp.sum(filtered_embeddings ** 2)
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        
        # Check that gradients exist and are not all zero
        assert grads is not None
        # Check that at least some gradients are non-zero
        total_grad_norm = jnp.sum(jnp.array([jnp.sum(g ** 2) for g in jax.tree_leaves(grads)]))
        assert total_grad_norm > 0
    
    def test_training_on_trivial_task(self):
        """Test that the filter can learn to solve a simple selection task"""
        batch_size, seq_len, embed_dim = 4, 8, 16
        learning_rate = 1  #0.01
        num_steps = 50
        
        # Create a trivial task: learn to select the first 2 tokens
        # We'll create embeddings where the first 2 tokens have a special pattern
        # and the rest are noise, then train the filter to select the first 2
        def create_batch_data(rng: jax.random.PRNGKey) -> jnp.ndarray:
            """Create batch data with clear pattern for first 2 tokens"""
            # First 2 tokens have high values in first dimension
            special_tokens = jnp.ones((batch_size, 2, embed_dim)) * 10.0
            # Rest are random noise
            noise_tokens = jax.random.normal(rng, (batch_size, seq_len - 2, embed_dim))
            return jnp.concatenate([special_tokens, noise_tokens], axis=1)
        
        # Initialize model
        filter_model = AdaptiveTokenFilter(hidden_dim=32, max_k=2)
        rng = jax.random.PRNGKey(42)
        token_embeddings = create_batch_data(rng)
        params = filter_model.init(rng, token_embeddings)
        
        # Define loss: encourage selection of first 2 tokens
        def loss_fn(params: dict, token_embeddings: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
            """Loss that rewards selecting first 2 tokens"""
            filtered_embeddings, selection_mask, _ = filter_model.apply(
                params, token_embeddings, rng=rng
            )
            
            # Target mask: first 2 tokens should be selected
            target_mask = jnp.zeros((batch_size, seq_len))
            target_mask = target_mask.at[:, :2].set(1.0)
            
            # Loss: MSE between actual and target selection
            selection_loss = jnp.mean((selection_mask - target_mask) ** 2)
            
            # Add regularization to prevent collapse
            reg_loss = 0.01 * jnp.mean(selection_mask ** 2)
            
            return selection_loss + reg_loss
        
        # Training loop
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        def train_step(params: dict, opt_state: dict, token_embeddings: jnp.ndarray, 
                      rng: jax.random.PRNGKey) -> Tuple[dict, dict, jnp.ndarray]:
            """Single training step"""
            loss_val, grads = jax.value_and_grad(loss_fn)(params, token_embeddings, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val
        
        # Train the model
        initial_loss = None
        final_loss = None
        
        for step in range(num_steps):
            rng, step_rng = jax.random.split(rng)
            token_embeddings = create_batch_data(step_rng)
            params, opt_state, loss_val = train_step(params, opt_state, token_embeddings, step_rng)
            
            if step == 0:
                initial_loss = float(loss_val)
            if step == num_steps - 1:
                final_loss = float(loss_val)
        
        # Test the trained model
        rng, test_rng = jax.random.split(rng)
        test_embeddings = create_batch_data(test_rng)
        _, selection_mask, _ = filter_model.apply(params, test_embeddings, rng=test_rng)
        
        # Check that the model learned to select the first 2 tokens
        # (on average, should select more from first 2 positions)
        first_two_selected = jnp.mean(selection_mask[:, :2])
        last_six_selected = jnp.mean(selection_mask[:, 2:])
        
        # Loss should decrease during training
        assert final_loss < initial_loss, \
            f"Loss didn't decrease: {initial_loss:.3f} -> {final_loss:.3f}"
        
        # The model should learn to prefer the first 2 tokens
        assert first_two_selected > last_six_selected, \
            f"Model didn't learn to prefer first 2 tokens: {first_two_selected:.3f} vs {last_six_selected:.3f}"
        
        # Final loss should be reasonably low
        assert final_loss < 0.5, f"Final loss too high: {final_loss:.3f}"
        
        # Check that selection mask has correct properties
        assert jnp.allclose(jnp.sum(selection_mask, axis=-1), 2.0, atol=1e-6), \
            "Selection mask should select exactly 2 tokens"
        assert jnp.all((selection_mask == 0) | (selection_mask == 1)), \
            "Selection mask should be binary"


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_sequence(self):
        """Test behavior with empty sequence (should handle gracefully)"""
        logits = jnp.array([[]])
        k = jnp.array([0])
        rng = jax.random.PRNGKey(42)
        
        result = gumbel_softmax_topk(logits, k, tau=1.0, hard=True, rng=rng)
        assert result.shape == (1, 0)
    
    def test_k_larger_than_sequence(self):
        """Test behavior when k is larger than sequence length"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        k = jnp.array([5])  # k > seq_len
        rng = jax.random.PRNGKey(42)
        
        result = gumbel_softmax_topk(logits, k, tau=1.0, hard=True, rng=rng)
        # Should select all available tokens
        assert jnp.all(result == 1)
    
    def test_negative_k(self):
        """Test behavior with negative k (should handle gracefully)"""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        k = jnp.array([-1])
        rng = jax.random.PRNGKey(42)
        
        result = gumbel_softmax_topk(logits, k, tau=1.0, hard=True, rng=rng)
        # Should select no tokens
        assert jnp.all(result == 0)
    
    def test_zero_temperature(self):
        """Test behavior with very low temperature"""
        logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        k = jnp.array([2])
        rng = jax.random.PRNGKey(42)
        
        # Very low temperature should still work
        result = gumbel_softmax_topk(logits, k, tau=1e-8, hard=False, rng=rng)
        assert jnp.allclose(result.sum(axis=-1), 1.0)
        assert jnp.all(result >= 0)


if __name__ == "__main__":
    pytest.main([__file__])
