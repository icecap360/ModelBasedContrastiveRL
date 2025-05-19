from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic
from utils.model_based import GCBilinearModelBasedValue, GCModelBasedActor, ModelBasedEncoder, compute_encoder_loss_core
import flax.linen as nn
from typing import Any, Sequence, Dict, Tuple # Added for typing
from flax.core import freeze, unfreeze

class CRLModelBasedAgent(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    CRL with DDPG+BC only fits a Q function, while CRL with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    # --- New attributes for the standalone ModelBasedEncoder ---
    encoder_module_def: ModelBasedEncoder = flax.struct.field(pytree_node=False) # The static module definition
    encoder: TrainState # TrainState for the online ModelBasedEncoder
    encoder_target_params: flax.core.FrozenDict # Parameters for the target ModelBasedEncoder
    # --- End of new attributes ---

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]

        obs, goals = batch['observations'], batch['value_goals']

        if module_name == 'critic':
            actions = batch['actions']
            v, phi, psi = self.network.select(module_name)(
                observations=batch['observations'],
                goals=batch['value_goals'],
                actions=actions,
                info=True,
                encoder_params=self.encoder_target_params, # Pass shared encoder's ONLINE parameters
                params=grad_params,
            )     
        else:
            actions = None
            v, phi, psi = self.network.apply_fn(
                {'params': grad_params}, # Pass all network params
                obs, goals, info=True, method_name=module_name
            )

        if not (module_name == 'critic' and self.config.get('critic_ensemble', True)): # Check if critic is ensembled via config
             # Add ensemble dim if not present, assuming non-ensembled output for value or non-ensembled critic
            if phi.ndim == 2 : phi = phi[jnp.newaxis, ...]
            if psi.ndim == 2 : psi = psi[jnp.newaxis, ...]
        # For ensembled critic, GCBV should already return (E, B, D)

        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        var = jnp.std(logits, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'variance': jnp.mean(var),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }
    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        # Maximize log Q if actor_log_q is True (which is default).
        if self.config['actor_log_q']:

            def value_transform(x):
                return jnp.log(jnp.maximum(x, 1e-6))
        else:

            def value_transform(x):
                return x
		
        dist = self.network.select('actor')(
            params= grad_params,
            observations= batch['observations'],
            goals=batch['actor_goals'],
            encoder_params=self.encoder_target_params, # Pass shared encoder's TARGET parameters
            )
        value_transform_fn = lambda x: jnp.log(jnp.maximum(x, 1e-6)) if self.config['actor_log_q'] else lambda x: x

        if self.config['actor_loss'] == 'awr':
            raise Exception('AWR loss is not supported yet.')
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

            q1, q2 = value_transform(self.network.select('critic')(
                observations=batch['observations'], 
                goals=batch['actor_goals'], actions=q_actions,
                encoder_params =self.encoder_target_params, # Shared TARGET encoder params
            ))
            q = jnp.minimum(q1, q2)
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()
            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if self.config['actor_loss'] == 'awr':
            value_loss, value_info = self.contrastive_loss(batch, grad_params, 'value')
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + value_loss + actor_loss
        return loss, info

    # @jax.jit
    # def update(self, batch: Dict): # This updates actor/critic/value (self.network)
    #     # First update (e.g., policy)
    #     new_self1, info = self.update_rl(batch)

    #     # Second update (e.g., value function), using updated RNG/network from first update
    #     new_self2, encoder_info = new_self1.update_encoder(batch)

    #     info.update(encoder_info)

    #     return new_self2, info

    @jax.jit
    def update(self, batch: Dict): # This updates actor/critic/value (self.network)
        new_rng, rng_for_total_loss = jax.random.split(self.rng)

        # network_grad_params are self.network.params
        # The loss_fn will be differentiated w.r.t these.
        def loss_fn_for_update(network_params_to_grad):
            return self.total_loss(batch, network_params_to_grad, rng=rng_for_total_loss)

        # `apply_loss_fn` is a helper that TrainState might have, or you implement it:
        # It computes grads and applies them.
        # Example:
        (loss_val, info), grads = jax.value_and_grad(loss_fn_for_update, has_aux=True)(self.network.params)
        new_network_state = self.network.apply_gradients(grads=grads)
        
        return self.replace(network=new_network_state, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        # observations = jnp.stack(jnp.split(observations, self.config['frame_stack'], axis=-1), 0)
        # goals = jnp.stack(jnp.split(goals, self.config['frame_stack'], axis=-1), 0)
        
        """Sample actions from the actor."""
        dist = self.network.select('actor')(
            observations, goals, temperature=temperature,
            encoder_params =self.encoder_target_params, # Pass shared encoder's TARGET parameters
            )
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        ex_batch=None
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        
        ex_goals = ex_observations

        rng = jax.random.PRNGKey(seed)
        rng, actor_critic_rng_main, encoder_rng_main = jax.random.split(rng, 3)

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            if config['actor_loss'] == 'awr':
                encoders['value_state'] = encoder_module()
                encoders['value_goal'] = encoder_module()

        # --- Initialize Standalone ModelBasedEncoder ---
        shared_encoder_module_def = ModelBasedEncoder(
            action_dim=action_dim, pixel_obs=config.get('pixel_obs_encoder', False),
            zs_dim=config.encoder_zs_dim, za_dim=config.get('encoder_za_dim', 256),
            zsa_dim=config.get('encoder_zsa_dim', 512), hdim=config.get('encoder_hdim', 512),
            activ_fn_name=config.get('encoder_activ_fn', 'elu'),
            state_feature_dim=ex_batch['stacked_observations'].shape[-1] if not config.get('pixel_obs_encoder', False) else 0,
            cnn_input_channels=ex_batch['stacked_observations'].shape[-1] if len(ex_batch['stacked_observations'].shape) == 4 and config.get('pixel_obs_encoder', False) else 3,
            cnn_flat_size=config.get('encoder_cnn_flat_size', 1568)
        )
        encoder_init_rng, _ = jax.random.split(encoder_rng_main) # Rng for encoder init
        dummy_state_for_enc_init = ex_batch['stacked_observations']
        dummy_action_for_enc_init = ex_batch['stacked_actions']
        initial_encoder_params = shared_encoder_module_def.init(
            encoder_init_rng, state_example=dummy_state_for_enc_init, 
            action_example=dummy_action_for_enc_init, method=ModelBasedEncoder._init_all_paths 
        )['params']
        encoder_optimizer = optax.adam(learning_rate=config.encoder_lr)
        encoder_train_state = TrainState.create(
            model_def=shared_encoder_module_def,
            params=initial_encoder_params, 
            tx=encoder_optimizer
        )
        encoder_target_params = flax.core.FrozenDict(initial_encoder_params)
        # --- End of standalone ModelBasedEncoder initialization ---
        # --- Define Actor/Critic/Value Networks ---
        # These are other encoders (e.g. visual) for actor/critic, not the shared model-based one.
        other_encoders_for_ac = {}
        if config.encoder is not None and encoder_modules_dict is not None:
            encoder_module_class = encoder_modules_dict[config.encoder]
            other_encoders_for_ac['critic_state'] = encoder_module_class()
            other_encoders_for_ac['critic_goal'] = encoder_module_class()
            # Assuming GCEncoder is defined and takes a concat_encoder
            other_encoders_for_ac['actor_gc_encoder'] = GCEncoder(concat_encoder=encoder_module_class()) 
            if config.actor_loss == 'awr':
                other_encoders_for_ac['value_state'] = encoder_module_class()
                other_encoders_for_ac['value_goal'] = encoder_module_class()
        
        # critic_def = GCBilinearValue(
        #         hidden_dims=config['value_hidden_dims'],
        #         latent_dim=config['latent_dim'],
        #         layer_norm=config['layer_norm'],
        #         ensemble=True,
        #         value_exp=True,
        #         state_encoder=encoders.get('critic_state'),
        #         goal_encoder=encoders.get('critic_goal'),
        #     )
        critic_def = GCBilinearModelBasedValue(
             hidden_dims=config.value_hidden_dims, latent_dim=config.latent_dim,
             encoder_module_def=shared_encoder_module_def, # Pass shared encoder DEFINITION
             layer_norm=config.layer_norm, ensemble=config.get('critic_ensemble', True), value_exp=True,
             state_encoder=other_encoders_for_ac.get('critic_state'), 
             goal_encoder=other_encoders_for_ac.get('critic_goal'),
        )
        if config.get('discrete', False):
            actor_def = GCDiscreteActor(hidden_dims=config.actor_hidden_dims, action_dim=action_dim, 
                                        gc_encoder=other_encoders_for_ac.get('actor_gc_encoder'))
        else:
            # actor_def = GCActor(hidden_dims=config.actor_hidden_dims, action_dim=action_dim, 
            #                     state_dependent_std=False, const_std=config.const_std, 
            #                     gc_encoder=other_encoders_for_ac.get('actor_gc_encoder'))
            actor_def = GCModelBasedActor(
                hidden_dims=config.actor_hidden_dims, 
                action_dim=action_dim, 
                state_dependent_std=False, const_std=config.const_std, 
                encoder_module_def=shared_encoder_module_def, # Pass shared encoder DEFINITION
                gc_encoder=other_encoders_for_ac.get('actor_gc_encoder'))

        network_components = {'critic': critic_def, 'actor': actor_def}
        
        # Arguments for initializing each component of self.network
        # Critic's __call__ needs: obs, goals, actions, encoder_params
        network_init_args = {
            'critic': (ex_observations, ex_goals, ex_actions, initial_encoder_params),
            'actor': (ex_observations, ex_goals, encoder_target_params), # Pass shared encoder's TARGET parameters
        }
        if config.actor_loss == 'awr':
            value_def = GCBilinearValue( # Assumed not to use shared model-based encoder
                hidden_dims=config.value_hidden_dims, latent_dim=config.latent_dim,
                layer_norm=config.layer_norm, ensemble=False, value_exp=True,
                state_encoder=other_encoders_for_ac.get('value_state'),
                goal_encoder=other_encoders_for_ac.get('value_goal'),
            )
            network_components['value'] = value_def
            network_init_args['value'] = (ex_observations, ex_goals)

        main_agent_module_dict = ModuleDict(network_components)
        main_network_params = main_agent_module_dict.init(actor_critic_rng_main, **network_init_args)['params']
        main_network_tx = optax.adam(learning_rate=config.lr)
        main_network_train_state = TrainState.create(
            model_def=main_agent_module_dict, params=main_network_params, tx=main_network_tx
        )
        
        return cls(
            rng=rng, network=main_network_train_state, 
            encoder_module_def=shared_encoder_module_def,
            encoder=encoder_train_state, encoder_target_params=encoder_target_params,
            config=flax.core.FrozenDict(config) if isinstance(config, ml_collections.ConfigDict) else config
        )

        # # Define value and actor networks.
        # if config['discrete']:
        #     critic_def = GCDiscreteBilinearCritic(
        #         hidden_dims=config['value_hidden_dims'],
        #         latent_dim=config['latent_dim'],
        #         layer_norm=config['layer_norm'],
        #         ensemble=True,
        #         value_exp=True,
        #         state_encoder=encoders.get('critic_state'),
        #         goal_encoder=encoders.get('critic_goal'),
        #         action_dim=action_dim,
        #     )
        # else:
        #     critic_def = GCBilinearModelBasedValue(
        #         hidden_dims=config['value_hidden_dims'],
        #         latent_dim=config['latent_dim'],
        #         layer_norm=config['layer_norm'],
        #         ensemble=True,
        #         value_exp=True,
        #         state_encoder=encoders.get('critic_state'),
        #         encoder_module_def=shared_encoder_module_def, # Pass shared encoder DEFINITION
        #         goal_encoder=encoders.get('critic_goal'),
        #     )

        # if config['actor_loss'] == 'awr':
        #     # AWR requires a separate V network to compute advantages (Q - V).
        #     value_def = GCBilinearValue(
        #         hidden_dims=config['value_hidden_dims'],
        #         latent_dim=config['latent_dim'],
        #         layer_norm=config['layer_norm'],
        #         ensemble=False,
        #         value_exp=True,
        #         state_encoder=encoders.get('value_state'),
        #         goal_encoder=encoders.get('value_goal'),
        #     )

        # if config['discrete']:
        #     actor_def = GCDiscreteActor(
        #         hidden_dims=config['actor_hidden_dims'],
        #         action_dim=action_dim,
        #         gc_encoder=encoders.get('actor'),
        #     )
        # else:
        #     actor_def = GCActor(
        #         hidden_dims=config['actor_hidden_dims'],
        #         action_dim=action_dim,
        #         state_dependent_std=False,
        #         const_std=config['const_std'],
        #         gc_encoder=encoders.get('actor'),
        #     )

        # network_info = dict(
        #     critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
        #     actor=(actor_def, (ex_observations, ex_goals)),
        # )
        # if config['actor_loss'] == 'awr':
        #     network_info.update(
        #         value=(value_def, (ex_observations, ex_goals)),
        #     )
        # networks = {k: v[0] for k, v in network_info.items()}
        # network_args = {k: v[1] for k, v in network_info.items()}

        # network_def = ModuleDict(networks)
        # network_tx = optax.adam(learning_rate=config['lr'])
        # network_params = network_def.init(init_rng, **network_args)['params']
        # network = TrainState.create(network_def, network_params, tx=network_tx)

        # return cls(rng, network=network, config=flax.core.FrozenDict(**config))

    # --- New method for computing encoder loss (wraps the core logic) ---
    def _encoder_loss_fn_for_grad(self, online_encoder_params: flax.core.FrozenDict, batch_for_encoder: dict):
        encoder_loss = compute_encoder_loss_core(
            online_encoder_params=online_encoder_params,
            target_encoder_params=self.encoder_target_params,
            encoder_module_def=self.encoder_module_def,
            states=batch_for_encoder['stacked_observations'], actions=batch_for_encoder['stacked_actions'],
            next_states=batch_for_encoder['stacked_next_observations'], not_done_mask=batch_for_encoder['masks'],
            enc_horizon=self.config['frame_stack'], 
            dyn_weight=self.config['dyn_weight'],
        )
        # loss_scale = jax.lax.stop_gradient(1.0 / (jnp.abs(encoder_loss).mean() + 1e-6))
        # encoder_loss = encoder_loss * loss_scale
        return encoder_loss

    @jax.jit
    def update_encoder(self, batch_for_encoder: dict):
        new_rng, _ = jax.random.split(self.rng)
        grad_fn = jax.value_and_grad(self._encoder_loss_fn_for_grad, argnums=0, has_aux=True)
        (loss, info), grads = grad_fn(self.encoder.params, batch_for_encoder)
        new_encoder_state = self.encoder.apply_gradients(grads=grads)
        return self.replace(encoder=new_encoder_state, rng=new_rng), info

    @jax.jit
    def update_encoder_target_hard(self):
        return self.replace(encoder_target_params=self.encoder.params)
    @jax.jit
    def update_encoder_target_soft(self):
        """
        Soft‐update the target encoder parameters via EMA:
            new_target = decay * old_target + (1 - decay) * online_params
        """
        decay: float = 0.999
        target_dict = unfreeze(self.encoder_target_params)
        online_dict = unfreeze(self.encoder.params)

        # 2. apply EMA per leaf
        new_target_dict = jax.tree_map(
            lambda tgt, src: decay * tgt + (1.0 - decay) * src,
            target_dict,
            online_dict,
        )

        # 3. turn back into FrozenDict
        new_target_params = freeze(new_target_dict)

        return self.replace(encoder_target_params=new_target_params)

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='crl_model_based',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            dyn_weight = 1.0,
            encoder_lr = 3e-4,
            encoder_zs_dim = 512,
            pixel_obs_encoder = False,
            encoder_za_dim = 256,
            encoder_zsa_dim = 512,
            encoder_hdim = 512,
            encoder_activ_fn = 'elu',
            encoder_cnn_flat_size = 1568,
        )
    )   
    return config

def masked_mse_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Computes Mean Squared Error on valid (masked) elements.
    predictions: (B, D) - Predictions for a single step in the horizon
    targets: (B, D) - Targets for a single step
    mask: (B,) - Mask for the step, 1.0 for valid, 0.0 for invalid.
    """
    if mask.ndim < predictions.ndim:
        mask = jnp.expand_dims(mask, axis=-1) # Broadcast mask to (B, 1)
    
    squared_error = jnp.square(predictions - targets) # (B, D)
    masked_squared_error = squared_error * mask       # (B, D)
    
    sum_loss = jnp.sum(masked_squared_error)
    num_elements = jnp.sum(mask) * predictions.shape[-1] # Count valid scalar elements
    
    return sum_loss / jnp.maximum(num_elements, 1e-8) # Avoid division by zero
