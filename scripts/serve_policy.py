"""
Policy serving script with wandb logging support for actual adaptive token filter statistics.

This script serves trained policies via websocket and optionally logs actual token usage
statistics from the adaptive token filter to wandb during evaluation.

Usage examples:
    # Basic serving without logging
    python scripts/serve_policy.py --env aloha_sim --port 8000
    
    # Serving with wandb logging enabled
    python scripts/serve_policy.py --env aloha_sim --port 8000 --wandb-enabled --wandb-project my-project
    
    # Serving with custom checkpoint and logging
    python scripts/serve_policy.py --policy checkpoint --policy.config pi0_aloha_sim --policy.dir /path/to/checkpoint --wandb-enabled

The wandb logging captures actual token usage for each inference by passing through
the ATF information from the model's embed_prefix method:
    - Per-camera token statistics (kept_frac, total_counts, expected_k)
    - ATF configuration parameters (expected_k, kept_fraction, random_k_prob)
    - Value function distribution information
    - Model configuration (atf_weight, atf_tau, atf_hidden_dim)
    - Training state (value_function_loss, training_step)
    - Inference metadata (inference count, timestamps)

Logs are sent to wandb with step=inference_count for tracking over time.
"""
import dataclasses
import enum
import logging
import socket
from typing import Any, Optional

import numpy as np
import tyro
import wandb

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    # Wandb logging configuration
    wandb_enabled: bool = False
    wandb_project: str = "openpi-policy-evaluation"
    wandb_run_name: Optional[str] = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


class TokenUsageLogger(_policy.BasePolicy):
    """Policy wrapper that logs actual token usage statistics from adaptive token filter to wandb."""
    
    def __init__(self, policy: _policy.BasePolicy, wandb_enabled: bool = True):
        self._policy = policy
        self._wandb_enabled = wandb_enabled
        self._inference_count = 0
        self._policy._model.AdaptiveTokenFilter.random_k_prob=0
        
    def infer(self, obs: dict) -> dict:
        """Run inference and log actual token usage statistics."""
        result = self._policy.infer(obs)
        
        if self._wandb_enabled:
            self._log_actual_token_usage_stats(result)
            
        if 'model_info' in result:
            del result['model_info']
        return result
    
    def _log_actual_token_usage_stats(self, result: dict) -> None:
        """Extract and log actual token usage statistics from the inference result."""
        self._inference_count += 1
        
        token_stats = {}
        
        # Extract ATF info from model_info if available
        if "model_info" in result:
            model_info = result["model_info"]
            
            # Log per-camera token statistics
            for key, value in model_info.items():
                if key.startswith("per_camera_pruning/"):
                    # Convert JAX arrays to Python scalars for wandb logging
                    if hasattr(value, 'item'):
                        token_stats[f"atf/actual/{key}"] = float(value.item())
                    else:
                        token_stats[f"atf/actual/{key}"] = float(value)
            
            # Log ATF configuration and training info
            for key, value in model_info.items():
                if key.startswith(("expected_k", "kept_fraction", "random_k_prob", "value_function_distribution")):
                    if hasattr(value, 'item'):
                        token_stats[f"atf/actual/{key}"] = float(value.item())
                    else:
                        token_stats[f"atf/actual/{key}"] = float(value)
        
        # Also log configuration parameters for context
        if hasattr(self._policy, '_model') and hasattr(self._policy._model, 'AdaptiveTokenFilter'):
            atf = self._policy._model.AdaptiveTokenFilter
            
            token_stats.update({
                'atf/config/use_value_function': bool(atf.use_value_function),
                'atf/config/random_k_prob': float(atf.random_k_prob),
                'atf/config/random_k_decay': float(atf.random_k_decay),
                'atf/config/max_tokens': int(atf.max_tokens),
                'atf/config/hidden_dim': int(atf.hidden_dim),
            })
            
            # Log training state if available
            if hasattr(atf, 'value_function_loss'):
                token_stats['atf/training/value_function_loss'] = float(atf.value_function_loss.value)
            if hasattr(atf, 'training_step'):
                token_stats['atf/training/training_step'] = int(atf.training_step.value)
        
        # Add inference metadata
        token_stats.update({
            'evaluation/inference_count': self._inference_count,
        })
        
        # Log to wandb
        wandb.log(token_stats, step=self._inference_count)
        
        # Log to console for debugging (only log every 10th inference to avoid spam)
        if token_stats and self._inference_count % 10 == 0:
            logging.info(f"Actual token usage stats (inference {self._inference_count}): {token_stats}")
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata from the wrapped policy."""
        return self._policy.metadata


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            policy = _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            policy = create_default_policy(args.env, default_prompt=args.default_prompt)
    
    # Wrap with token usage logger if wandb is enabled
    if args.wandb_enabled:
        policy = TokenUsageLogger(policy, wandb_enabled=True)
    
    return policy


def main(args: Args) -> None:
    # Initialize wandb if enabled
    if args.wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"policy-eval-{args.env.value}",
            config={
                "env": args.env.value,
                "port": args.port,
                "record": args.record,
                "default_prompt": args.default_prompt,
                "policy_type": type(args.policy).__name__,
            }
        )
        logging.info(f"Initialized wandb logging for project: {args.wandb_project}")
    
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    finally:
        if args.wandb_enabled:
            wandb.finish()
            logging.info("Wandb logging finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
