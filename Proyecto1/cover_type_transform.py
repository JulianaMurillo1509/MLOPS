
import tensorflow as tf
import tensorflow_transform as tft

import cover_type_constants

# Unpack the contents of the constants module
_SCALE_TO_ONE_FEATURE_KEYS = cover_type_constants.SCALE_TO_ONE_FEATURE_KEYS
_SCALE_BE_MIN_MAX_FEATURE_KEYS=cover_type_constants.SCALE_BE_MIN_MAX_FEATURE_KEYS
_SCALE_TO_Z_SCORE_FEATURE_KEYS=cover_type_constants.SCALE_TO_Z_SCORE_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = cover_type_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = cover_type_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = cover_type_constants.FEATURE_BUCKET_COUNT
_LABEL_KEY = cover_type_constants.LABEL_KEY
_transformed_name = cover_type_constants.transformed_name


# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # Scale these features to the range [0,1]
    for key in _SCALE_TO_ONE_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(
            inputs[key])
        
    # Scale these features to the range [0,1]
    for key in _SCALE_BE_MIN_MAX_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_by_min_max(
            inputs[key],output_min= 0.0,
            output_max=1.0,)   
        
    # Scale these features to the range [0,1]
    for key in _SCALE_TO_Z_SCORE_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_z_score(
            inputs[key])
        
    
    # Bucketize these features
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            inputs[key], _FEATURE_BUCKET_COUNT[key])

    # Convert strings to indices in a vocabulary
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

    # Convert the label strings to an index
    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(inputs[_LABEL_KEY])

    return outputs
