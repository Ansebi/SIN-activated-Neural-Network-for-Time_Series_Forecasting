import logging
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Add


def build_sin_model(
    x, y,
    wavelen=365.25,
    init_x_shift=1,
    init_y_shift='auto',
    init_amplitude='auto',
    learning_rate=10**-1,
    linear_trend=True,
    sin_components_trainable=True,
    trainability_map = None,
    model_name = None,
    show_summary=True
):
  """
  Sin-wave model with linear component\n\n

  example_trainability_map = {\n
    'linear_rotation': True,\n
    'linear_rotation_handler': True,\n
    'y_shift': True,\n
    'y_shift_amplifier': True,\n
    'frequency': False,\n
    'phase_shift': True,\n
    'phase_shift_amplifier': True,\n
    'sin': False,\n
    'amplitude': True,\n
    'output': False\n
    }
  """

  if not model_name:
    model_name = 'Sin-wave_model_with_linear_component'

  default_trainability_map = {
    'linear_rotation': True,
    'linear_rotation_handler': True,
    'y_shift': True,
    'y_shift_amplifier': True,
    'frequency': False,
    'phase_shift': True,
    'phase_shift_amplifier': True,
    'sin': False,
    'amplitude': True,
    'output': False
    }

  if not trainability_map:
    trainability_map = default_trainability_map

  if init_amplitude == 'auto':
    init_amplitude = (y.max() - y.min()) / 2

  if init_y_shift == 'auto':
    init_y_shift = y.mean()

  initial_linear_rotation = 10**-6
  adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  input_layer = Input(
      name='input',
      shape=[x.shape[1]]
      )

  frequency_layer = Dense(
      1,
      name='frequency',
      kernel_initializer=tf.keras.initializers.Constant(value=2*np.pi/wavelen),
      use_bias=False
      )(input_layer)

  phase_shift_layer = Dense(
      1,
      name='phase_shift',
      activation= lambda x: 0 * x + init_x_shift,
      use_bias=False,
      kernel_initializer=tf.keras.initializers.Constant(value=init_x_shift)
      )(input_layer)

  phase_shift_amplifier_layer = Dense(
      1,
      name='phase_shift_amplifier',
      use_bias=False,
      kernel_initializer=tf.keras.initializers.Constant(value=1)
      )(phase_shift_layer)

  sin_input_layer = Add(name='sin_input')(
      [
          frequency_layer,
          phase_shift_amplifier_layer
       ])

  sin_layer = Dense(
      1,
      name='sin',
      activation=lambda sin_input: tf.math.sin(sin_input),
      kernel_initializer=tf.keras.initializers.Constant(value=1),
      use_bias=False
      )(sin_input_layer)

  sin_amplitude_layer = Dense(
      1,
      name='amplitude',
      kernel_initializer=tf.keras.initializers.Constant(value=init_amplitude),
      use_bias=False
      )(sin_layer)

  linear_rotation_layer = Dense(
      1,
      name='linear_rotation',
      kernel_initializer=tf.keras.initializers.Constant(value=initial_linear_rotation),
      use_bias=False
      )(input_layer)

  linear_rotation_handler_layer = Dense(
      1,
      name='linear_rotation_handler',
      kernel_initializer=tf.keras.initializers.Constant(value=initial_linear_rotation),
      use_bias=False
      )(linear_rotation_layer)

  y_shift_layer = Dense(
      1,
      name='y_shift',
      activation= lambda x: 0 * x + init_y_shift,
      use_bias=False,
      kernel_initializer=tf.keras.initializers.Constant(value=init_y_shift)
      )(input_layer)

  y_shift_amplifier_layer = Dense(
      1,
      name='y_shift_amplifier',
      kernel_initializer=tf.keras.initializers.Constant(value=1),
      use_bias=False,
      )(y_shift_layer)

  linear_component_layer = Add(name='linear_component')([
      linear_rotation_handler_layer,
      y_shift_amplifier_layer
      ])

  output_layer = Add(name='output')([
      linear_component_layer,
      sin_amplitude_layer
      ])

  model = Model(inputs=input_layer, outputs=output_layer, name=model_name)

  for layer_name, trainability in trainability_map.items():
    model.get_layer(name=layer_name).trainable = trainability

  linear_components = [
      'linear_rotation',
      'linear_rotation_handler',
      'y_shift',
      'y_shift_amplifier'
      ]

  sin_components = [
      'frequency',
      'phase_shift',
      'phase_shift_amplifier',
      'sin',
      'amplitude'
      ]

  if not linear_trend:
    for layer_name in linear_components:
      model.get_layer(name=layer_name).trainable = False

  if not sin_components_trainable:
    for layer_name in sin_components:
      model.get_layer(name=layer_name).trainable = False

  model.compile(optimizer=adam, loss='mse')
  if show_summary:
    model.summary()
  return model


def build_multiwave_model(x, y, waves: list, learning_rate=10**-1, wavelen=365.25):
      """
      waves: (list) - are fractions (or whole numbers) of the base wavelen to
      produce additional waves with diverse frequencies.
      If you have now idea, which wavelen fractions to give,
      use auto_n_relative_wavelengths() function to generate the example 'waves'.
      """
      adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)
      wave_models = {}
      input_layer = tf.keras.layers.Input(
            name='input',
            shape=[x.shape[1]]
            )
      for i in waves:
            wave_models[i] = build_sin_model(
                  x=x,
                  y=y,
                  wavelen=wavelen*i,
                  model_name=f'Wave_{round(i, 3)}',
                  show_summary=False
                  )(input_layer)
      output_layer = tf.keras.layers.Add(name='output')(list(wave_models.values()))
      multiwave_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
      multiwave_model.compile(optimizer=adam, loss='mse')
      multiwave_model.summary()
      return multiwave_model


def auto_n_relative_wavelengths(n: int):
    if n == 1:
        return [1]
    even = 0
    if not n % 2:
        warning = 'It is preferable to have odd number of waves'
        warning += ' so that 1 (single wave) in the middle is also returned.'
        logging.warning(warning)
        even = 1
    waves = list(np.ones(n))
    for i in list(range(1, n // 2 + 1)):
        waves[n // 2 + i - even] = i + 1
        waves[n // 2 - i] = 1 / (i + 1)
    return waves