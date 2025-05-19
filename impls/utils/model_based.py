import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers as flax_initializers
from typing import Callable, Sequence, Any, Optional
from functools import partial
from .networks import MLP, ensemblize, default_init
PRNGKey = Any
import flax
from optax import l2_loss
import distrax

default_kernel_init = flax_initializers.variance_scaling(scale=2.0, mode='fan_avg', distribution='uniform')
default_bias_init = flax_initializers.zeros

class LnActiv(nn.Module):
    activation: Callable = nn.gelu
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # It's good practice to name submodules for clarity in parameter trees, though not strictly necessary here.
        x = nn.LayerNorm(epsilon=self.eps, use_bias=True, use_scale=True, name="ln")(x)
        return self.activation(x)

class ModelBasedMLP(nn.Module):
    output_dim: int
    hdim: int
    activ_fn_name: str = 'gelu'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activ_fn = getattr(nn, self.activ_fn_name)
        y = nn.Dense(features=self.hdim, kernel_init=default_kernel_init, bias_init=default_bias_init)(x)
        y = LnActiv(activation=activ_fn)(y) # This is fine as BaseMLP.__call__ is @nn.compact
        y = nn.Dense(features=self.hdim, kernel_init=default_kernel_init, bias_init=default_bias_init)(y)
        y = LnActiv(activation=activ_fn)(y) # This is fine as BaseMLP.__call__ is @nn.compact
        y = nn.Dense(features=self.output_dim, kernel_init=default_kernel_init, bias_init=default_bias_init)(y)
        return y

class ModelBasedEncoder(nn.Module):
    action_dim: int
    pixel_obs: bool
    num_bins: int = 65
    zs_dim: int = 512
    za_dim: int = 256
    zsa_dim: int = 512
    hdim: int = 512
    activ_fn_name: str = 'gelu'
    state_feature_dim: int = 0
    cnn_input_channels: int = 3
    cnn_flat_size: int = 1568

    def setup(self):
        self.activ_fn = getattr(nn, self.activ_fn_name)
        
        if self.pixel_obs:
            self._zs_cnn1 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), kernel_init=default_kernel_init, bias_init=default_bias_init)
            self._zs_cnn2 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), kernel_init=default_kernel_init, bias_init=default_bias_init)
            self._zs_cnn3 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), kernel_init=default_kernel_init, bias_init=default_bias_init)
            self._zs_cnn4 = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), kernel_init=default_kernel_init, bias_init=default_bias_init)
            self._zs_lin = nn.Dense(features=self.zs_dim, kernel_init=default_kernel_init, bias_init=default_bias_init)
            self.zs_encoder_fn = self._cnn_zs
        else:
            self._zs_mlp = ModelBasedMLP(output_dim=self.zs_dim, hdim=self.hdim, activ_fn_name=self.activ_fn_name)
            self.zs_encoder_fn = self._mlp_zs

        self._za_lin = nn.Dense(features=self.za_dim, kernel_init=default_kernel_init, bias_init=default_bias_init)
        self._zsa_mlp = ModelBasedMLP(output_dim=self.zsa_dim, hdim=self.hdim, activ_fn_name=self.activ_fn_name)
        self._output_next_zs = nn.Dense(features=self.zs_dim, kernel_init=default_kernel_init, bias_init=default_bias_init)

    @nn.compact  # <--- ADDED @nn.compact
    def _cnn_zs(self, state: jnp.ndarray) -> jnp.ndarray:
        zs = state / 255.0 - 0.5
        zs = self.activ_fn(self._zs_cnn1(zs))
        zs = self.activ_fn(self._zs_cnn2(zs))
        zs = self.activ_fn(self._zs_cnn3(zs))
        zs = self.activ_fn(self._zs_cnn4(zs))
        zs = zs.reshape((zs.shape[0], -1))
        if zs.shape[1] != self.cnn_flat_size and self.pixel_obs:
            raise ValueError(f"CNN output flattened size {zs.shape[1]} does not match expected cnn_flat_size {self.cnn_flat_size}")
        zs = self._zs_lin(zs)
        # This LnActiv is now correctly defined within a @compact method
        return LnActiv(activation=self.activ_fn, name="final_zs_ln_activ_cnn")(zs)

    @nn.compact  # <--- ADDED @nn.compact
    def _mlp_zs(self, state: jnp.ndarray) -> jnp.ndarray:
        zs_encoded = self._zs_mlp(state) # self._zs_mlp is an instance of BaseMLP from setup()
        # This LnActiv is now correctly defined within a @compact method
        return LnActiv(activation=self.activ_fn, name="final_zs_ln_activ_mlp")(zs_encoded)

    def encode_state(self, state: jnp.ndarray) -> jnp.ndarray:
        zs = self.zs_encoder_fn(state)
        # norm = jnp.linalg.norm(zs, axis=-1, keepdims=True) + 1e-6
        # zs_normalized = zs / norm
        return zs

    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        za = self.activ_fn(self._za_lin(action))
        zsa_input = jnp.concatenate([zs, za], axis=-1)
        return self._zsa_mlp(zsa_input)

    def model_all(self, zs: jnp.ndarray, action: jnp.ndarray):
        zsa = self.__call__(zs, action)
        next_zs_delta = self._output_next_zs(zsa)
        return next_zs_delta

    def _init_all_paths(self, state_example: jnp.ndarray, action_example: jnp.ndarray):
        zs_example = self.encode_state(state_example)
        self.model_all(zs_example, action_example)
        return zs_example

class GCModelBasedActor(nn.Module):
    """Goal-conditioned actor.

    Attributes: 
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    encoder_module_def: ModelBasedEncoder # The static definition of the shared encoder
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        encoder_params: flax.core.FrozenDict= None, # Parameters of the shared encoder
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            # --- Use the shared ModelBasedEncoder ---
            observations = self.encoder_module_def.apply(
                {'params': encoder_params}, observations, method=ModelBasedEncoder.encode_state
            )
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution

# --- EDITED GCBilinearModelBasedValue (Critic) ---
class GCBilinearModelBasedValue(nn.Module):
    hidden_dims: Sequence[int]
    latent_dim: int
    encoder_module_def: ModelBasedEncoder # The static definition of the shared encoder
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None # For other, non-shared state encodings (e.g., visual pre-encoder)
    goal_encoder: nn.Module = None  # For other, non-shared goal encodings
    activ_fn_name: str = 'gelu'

    def setup(self):
        mlp_module = MLP
        output_dim_phi_psi = self.latent_dim
        mlp_module = ensemblize(mlp_module, 2)

        self.phi_mlp = mlp_module((*self.hidden_dims, output_dim_phi_psi), activate_final=False, layer_norm=self.layer_norm)
        self.psi_mlp = mlp_module((*self.hidden_dims, output_dim_phi_psi), activate_final=False, layer_norm=self.layer_norm)
        activ_fn = getattr(nn, self.activ_fn_name)
        self.norm = LnActiv(activation=activ_fn) # This is fine as BaseMLP.__call__ is @nn.compact
        self.norm_goals = LnActiv(activation=activ_fn)# This is fine as BaseMLP.__call__ is @nn.compact


    def __call__(self, 
                 observations: jnp.ndarray, 
                 goals: jnp.ndarray, 
                 actions: jnp.ndarray, 
                 encoder_params: flax.core.FrozenDict, # Parameters of the shared encoder
                 info: bool = False):

        if self.state_encoder is not None: # Apply pre-encoder for observations if any
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None: # Apply pre-encoder for goals if any
            goals = self.goal_encoder(goals)

        # --- Use the shared ModelBasedEncoder ---
        zs = self.encoder_module_def.apply(
            {'params': encoder_params}, observations, method=ModelBasedEncoder.encode_state
        )
        zsa = self.encoder_module_def.apply(
            {'params': encoder_params}, zs, actions, method=ModelBasedEncoder.__call__
        )
        
        zsa = self.norm(zsa)

        zs_goals = self.encoder_module_def.apply(
            {'params': encoder_params}, goals, method=ModelBasedEncoder.encode_state
        )
        zs_goals = self.norm_goals(zs_goals)
        # zs_goals = goals

        phi_output_raw = self.phi_mlp(zsa) 
        psi_output_raw = self.psi_mlp(zs_goals)

        num_ensembles = 2 if self.ensemble else 1
        if self.ensemble:
            # Reshape to (num_ensembles, batch, latent_dim)
            phi_output = phi_output_raw.reshape(-1, num_ensembles, self.latent_dim).transpose(1,0,2)
            psi_output = psi_output_raw.reshape(-1, num_ensembles, self.latent_dim).transpose(1,0,2)
        else:
            # Add ensemble dim: (1, batch, latent_dim)
            phi_output = phi_output_raw[jnp.newaxis, ...] 
            psi_output = psi_output_raw[jnp.newaxis, ...]


        # einsum for ensemble: 'eik,ejk->ije' where e=ensemble, i=batch1, j=batch2, k=latent
        # Here, we do element-wise product for V(s,g) = sum(phi(s,a)_e * psi(g)_e) / sqrt(d) for each ensemble member
        # Then sum over latent_dim. Output shape should be (num_ensembles, batch_size)
        v = jnp.sum(phi_output * psi_output, axis=-1) / jnp.sqrt(self.latent_dim)
        
        if not self.ensemble: # Squeeze ensemble dim if not used
            v = v.squeeze(axis=0)
            phi_output = phi_output.squeeze(axis=0)
            psi_output = psi_output.squeeze(axis=0)


        if self.value_exp: v = jnp.exp(v)
        return (v, phi_output, psi_output) if info else v
        
def compute_encoder_loss_core(
    online_encoder_params: flax.core.FrozenDict, target_encoder_params: flax.core.FrozenDict,
    encoder_module_def: ModelBasedEncoder, states: jnp.ndarray, actions: jnp.ndarray,
    next_states: jnp.ndarray, not_done_mask: jnp.ndarray, enc_horizon: int, dyn_weight: float,
):
    batch_size = states.shape[0]; zs_dim = encoder_module_def.zs_dim; state_shape = states.shape[2:]
    # flat_next_states = next_states.reshape(-1, *state_shape) 
    target_zs_horizon = encoder_module_def.apply(
        {'params': target_encoder_params}, next_states, method=ModelBasedEncoder.encode_state
    )
    initial_states_for_online_encoder = states[:, 0] 
    current_pred_zs = encoder_module_def.apply(
        {'params': online_encoder_params}, initial_states_for_online_encoder, method=ModelBasedEncoder.encode_state
    )
    def loop_body(carry_pred_zs, i):
        action_step = actions[:, i]; target_zs_step = target_zs_horizon[:, i]
        next_predicted_latent_state = encoder_module_def.apply(
            {'params': online_encoder_params}, carry_pred_zs, action_step, method=ModelBasedEncoder.model_all 
        )
        step_dyn_loss = jnp.mean(jnp.square(next_predicted_latent_state - target_zs_step))
        return next_predicted_latent_state, step_dyn_loss
    final_pred_zs, dyn_losses_per_step = jax.lax.scan(loop_body, current_pred_zs, jnp.arange(enc_horizon))
    total_encoder_loss = jnp.sum(dyn_losses_per_step) * dyn_weight 
    avg_dyn_loss_per_step = jnp.mean(dyn_losses_per_step)
    return total_encoder_loss, {'encoder_loss_total': total_encoder_loss, 'encoder_avg_step_mse': avg_dyn_loss_per_step}



def sample_gumbel(key, shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    u = jax.random.uniform(key, shape=shape, minval=0, maxval=1)
    return -jnp.log(-jnp.log(u + eps) + eps)

def gumbel_softmax_sample(logits, tau, key):
    """Draw a sample from the Gumbel-Softmax distribution"""
    gumbel_noise = sample_gumbel(key, logits.shape)
    y = logits + gumbel_noise
    return jax.nn.softmax(y / tau, axis=-1)

def gumbel_softmax(logits, tau=1.0, key=None, hard=False):
    """Gumbel-Softmax: differentiable sampling from a categorical"""
    y_soft = gumbel_softmax_sample(logits, tau, key)
    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1])
        # Straight-through estimator
        y = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
    else:
        y = y_soft
    return y

# Example Usage (no changes needed here for the fix)
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    key_params_mlp_base_init, key_params_encoder_mlp, key_params_encoder_cnn, \
    key_params_policy_d, key_params_policy_c, key_params_value, key_gumbel = jax.random.split(key, 7) # Adjusted key split

    # --- BaseMLP Example ---
    print("--- BaseMLP Example ---")
    base_mlp = ModelBasedMLP(output_dim=10, hdim=64)
    dummy_input_mlp_base = jnp.ones((1, 32))
    params_mlp_base = base_mlp.init(key_params_mlp_base_init, dummy_input_mlp_base)
    output_mlp_base = base_mlp.apply(params_mlp_base, dummy_input_mlp_base)
    print("BaseMLP Output Shape:", output_mlp_base.shape)

    # --- ModelBasedEncoder Example (MLP) ---
    print("\n--- ModelBasedEncoder Example (MLP) ---")
    encoder_mlp = ModelBasedEncoder(action_dim=4, pixel_obs=False, state_feature_dim=128, 
                          zs_dim=256, za_dim=64, zsa_dim=256, hdim=128)
    dummy_state_mlp = jnp.ones((2, 128))
    dummy_action_enc = jnp.ones((2, 4))
    
    variables_encoder_mlp = encoder_mlp.init(
        key_params_encoder_mlp, dummy_state_mlp, dummy_action_enc, method=ModelBasedEncoder._init_all_paths
    )

    zs_from_mlp_state = encoder_mlp.apply(variables_encoder_mlp, dummy_state_mlp, method=encoder_mlp.encode_state)
    print("zs from MLP state shape:", zs_from_mlp_state.shape)

    zsa_output_mlp = encoder_mlp.apply(variables_encoder_mlp, zs_from_mlp_state, dummy_action_enc)
    print("ModelBasedEncoder (MLP) zsa output shape:", zsa_output_mlp.shape)
    
    done_mlp, next_zs_delta_mlp, reward_logits_mlp = encoder_mlp.apply(
        variables_encoder_mlp, zs_from_mlp_state, dummy_action_enc, method=encoder_mlp.model_all
    )
    print("ModelBasedEncoder (MLP) done shape:", done_mlp.shape)
    print("ModelBasedEncoder (MLP) next_zs_delta shape:", next_zs_delta_mlp.shape)
    print("ModelBasedEncoder (MLP) reward_logits shape:", reward_logits_mlp.shape)


    print("\n--- ModelBasedEncoder Example (CNN) ---")
    encoder_cnn = ModelBasedEncoder(action_dim=4, pixel_obs=True, cnn_input_channels=3, 
                          zs_dim=256, za_dim=64, zsa_dim=256, hdim=128,
                          cnn_flat_size=8192)
    dummy_state_cnn = jnp.ones((2, 128, 128, 3))
    
    variables_encoder_cnn = encoder_cnn.init(
        key_params_encoder_cnn, dummy_state_cnn, dummy_action_enc, method=ModelBasedModelBasedEncoder._init_all_paths
    )

    zs_from_cnn_state = encoder_cnn.apply(variables_encoder_cnn, dummy_state_cnn, method=encoder_cnn.encode_state)
    print("zs from CNN state shape:", zs_from_cnn_state.shape)

    zsa_output_cnn = encoder_cnn.apply(variables_encoder_cnn, zs_from_cnn_state, dummy_action_enc)
    print("ModelBasedModelBasedEncoder (CNN) zsa output shape:", zsa_output_cnn.shape)

    done_cnn, next_zs_delta_cnn, reward_logits_cnn = encoder_cnn.apply(
        variables_encoder_cnn, zs_from_cnn_state, dummy_action_enc, method=encoder_cnn.model_all
    )
    print("ModelBasedModelBasedEncoder (CNN) done shape:", done_cnn.shape)
    print("ModelBasedModelBasedEncoder (CNN) next_zs_delta shape:", next_zs_delta_cnn.shape)
    print("ModelBasedModelBasedEncoder (CNN) reward_logits shape:", reward_logits_cnn.shape)


    print("\n--- Policy Example (Discrete) ---")
    policy_discrete = Policy(action_dim=5, discrete=True, hdim=128)
    dummy_zs_policy = zs_from_cnn_state # Using output from CNN encoder example

    variables_policy_discrete = policy_discrete.init(key_params_policy_d, dummy_zs_policy, training=True, rng_key=key_gumbel)
    
    action_discrete, pre_activ_discrete = policy_discrete.apply(
        variables_policy_discrete, dummy_zs_policy, training=True, rng_key=key_gumbel
    )
    print("Policy (Discrete) Training Action shape:", action_discrete.shape)
    print("Policy (Discrete) Training Pre-activation shape:", pre_activ_discrete.shape)

    eval_action_discrete, _ = policy_discrete.apply(
        variables_policy_discrete, dummy_zs_policy, training=False # No rng_key needed if training=False and Gumbel is not used
    )
    print("Policy (Discrete) Eval Action shape (Softmax):", eval_action_discrete.shape)
    
    act_eval_action_discrete = policy_discrete.apply(
        variables_policy_discrete, dummy_zs_policy, training=False, method=policy_discrete.act
    )
    print("Policy (Discrete) 'act' (training=False) shape:", act_eval_action_discrete.shape)


    print("\n--- Policy Example (Continuous) ---")
    policy_continuous = Policy(action_dim=3, discrete=False, hdim=128)
    variables_policy_continuous = policy_continuous.init(key_params_policy_c, dummy_zs_policy)
    
    action_continuous, pre_activ_continuous = policy_continuous.apply(variables_policy_continuous, dummy_zs_policy)
    print("Policy (Continuous) Action shape:", action_continuous.shape)
    print("Policy (Continuous) Pre-activation shape:", pre_activ_continuous.shape)

    print("\n--- Value Function Example ---")
    value_fn = Value(hdim=128)
    dummy_zsa_value = zsa_output_cnn # Using output from CNN encoder example
    variables_value = value_fn.init(key_params_value, dummy_zsa_value)
    
    q_values = value_fn.apply(variables_value, dummy_zsa_value)
    print("Value Function Output (Q1, Q2) shape:", q_values.shape)