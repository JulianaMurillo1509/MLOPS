
# Features with string data types that will be converted to indices
#CATEGORICAL_FEATURE_KEYS = [
#    'education', 'marital-status', 'occupation', 'race', 'relationship', 'workclass', 'sex', 'native-country'
#]
CATEGORICAL_FEATURE_KEYS = ['Wilderness_Area','Soil_Type']
SCALE_TO_ONE_FEATURE_KEYS = ['Elevation','Slope']

SCALE_BE_MIN_MAX_FEATURE_KEYS = ['Horizontal_Distance_To_Hydrology']

SCALE_TO_Z_SCORE_FEATURE_KEYS = ['Horizontal_Distance_To_Fire_Points','Hillshade_9am','Hillshade_Noon','Horizontal_Distance_To_Roadways']

# Feature that can be grouped into buckets
#BUCKET_FEATURE_KEYS = ['age']
BUCKET_FEATURE_KEYS =[]
# Number of buckets used by tf.transform for encoding each bucket feature.
#FEATURE_BUCKET_COUNT = {'age': 4}
FEATURE_BUCKET_COUNT={}
# Feature that the model will predict
LABEL_KEY = 'Cover_Type'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
