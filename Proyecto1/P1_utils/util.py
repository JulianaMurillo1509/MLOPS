import tensorflow as tf
import pandas as pd
from google.protobuf.json_format import MessageToDict
from typing import List,Text
from ml_metadata import proto
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2
from ml_metadata.proto import metadata_store_service_pb2_grpc

def get_records(dataset, num_records):
    '''Extracts records from the given dataset.
    Args:
        dataset (TFRecordDataset): dataset saved by ExampleGen
        num_records (int): number of records to preview
    '''
    
    # initialize an empty list
    records = []
    
    # Use the `take()` method to specify how many records to get
    for tfrecord in dataset.take(num_records):
        
        # Get the numpy property of the tensor
        serialized_example = tfrecord.numpy()
        
        # Initialize a `tf.train.Example()` to read the serialized data
        example = tf.train.Example()
        
        # Read the example data (output is a protocol buffer message)
        example.ParseFromString(serialized_example)
        
        # convert the protocol bufffer message to a Python dictionary
        example_dict = (MessageToDict(example))
        
        # append to the records list
        records.append(example_dict)
        
    return records

def display_types(types):
    # Helper function to render dataframes for the artifact and execution types
    table = {'id': [], 'name': []}
    for a_type in types:
        table['id'].append(a_type.id)
        table['name'].append(a_type.name)
    return pd.DataFrame(data=table)

def display_artifacts(store, artifacts, base_dir):
    # Helper function to render dataframes for the input artifacts
    table = {'artifact id': [], 'type': [], 'uri': []}
    for a in artifacts:
        table['artifact id'].append(a.id)
        artifact_type = store.get_artifact_types_by_id([a.type_id])[0]
        table['type'].append(artifact_type.name)
        table['uri'].append(a.uri.replace(base_dir, './'))
    return pd.DataFrame(data=table)

def display_properties(store, node):
    # Helper function to render dataframes for artifact and execution properties
    table = {'property': [], 'value': []}
    
    for k, v in node.properties.items():
        table['property'].append(k)
        table['value'].append(
            v.string_value if v.HasField('string_value') else v.int_value)
    
    for k, v in node.custom_properties.items():
        table['property'].append(k)
        table['value'].append(
            v.string_value if v.HasField('string_value') else v.int_value)
    
    return pd.DataFrame(data=table)

def follow_artifacts(store,artifacts):
    table = {'properties': [], 'id': [], 'name': [], 'uri': []}
    
    for a in artifacts:
        table['id'].append(a.id)
        table['name'].append(a.name)
        table['uri'].append(a.uri)
        table['properties'].append(a.custom_properties)
        
    
    return pd.DataFrame(data=table)

def followArtifacts2(store,artifacts,uri):
    table = {'artifact id': [], 'type': [], 'uri': []}
    
    ids_Transform = [o.id for o in artifacts]
    execution_events_Transform = store.get_events_by_execution_ids(ids_Transform)
    ids_Artifacts_Transform = [o.artifact_id for o in execution_events_Transform]
    list_artifacts_Transform= store.get_artifacts_by_id(ids_Artifacts_Transform)
    result = display_artifacts(store,list_artifacts_Transform,uri)

  
    return result
    