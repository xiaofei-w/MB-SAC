# Documentation

* [Dynamics Model](#dynamicsmodel)
    * Ensemble-MLP
    * RNN
    
## Dynamics Model
Inheritance Relationship:
* [`class PredictEnv`](#predictenv)
    * [`class PredictEnvRNN`](#predictenvrnn)
* [`class Model`](#model)
    
### PredictEnv
Highest level of abstraction of the model that acts like a gym environment and accounts for all interaction with the agent and the runner

Attributes: `model, env_name`

`def step(self, state, action, deterministic=False)->next_state, reward, done, info`

`def train(self, data, batch_size, steps, logger)->loss`
       
### PredictEnvRNN
Inherits PredictEnv, with small change in dimension to cast RNN
Attributes: `model, env_name`

`def step(self, states, actions, deterministic=False)->next_state, reward, done, info`
where states, actions are of shape (seq_len, batch, \*) or (seq_len, \*)

`def train(self, replay_buffer, batch_size, steps, logger)->loss`

### Model
Interface for a low-level nn.Module dynamics model

`def __init__(self, state_dim, act_dim)`

`def forward`

`def loss`

`def update`

`def predict(self, action, obs)`

`def set_train_mode(self)`

`def set_eval_mode(self)`

## Agent

## Agent-Model Interaction

