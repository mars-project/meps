# Motivation
Shuffle is the core operation for large-scale data processing. The data is partitioned according to the partition key in each record through the ALL-TO-ALL connection, and sent from all mappers to all reducers. Due to the nature of its ALL-TO_ALL connections, shuffle often becomes the bottleneck of the performance, stability and scalability of the entire system. This MEP introduces a pull-based shuffle implementation using ray future which serves following goals:
- Improve Mars-on-Ray shuffle scalability by reducing shuffle meta
- Improve scheduling performance by using ray c++ scheduler
- Support shuffle failover by lineage recontruction
- Setup the foundation for push-based shuffle using ray future

## Design and Architecture
When the Mars subtask has upstream and downstream dependencies, the downstream subtask's dependence on the output of the upstream subtask is determined by the data key. The input data required for subtask execution is a context dict object from data key to data. 

For non-shuffle operators, the data key is the chunk key; but for shuffle operators, the data key is a tuple of (chunk key, reducer index). 

Therefore, in order to build a Shuffle DAG based on Ray future, either all the data keys output by each mapper subtask are known in advance, or the data keys are removed.

### Why not using data key?
If data keys are used, all the data keys output by each mapper subtask must be known in advance, then pack those data keys and data into a dict and pass it to the subtask when the subtask is submitted.

A new `get_data_keys()` interface needs to be added to all shuffle operators to obtain all the data keys of the subtask in advance:
- For non-shuffle operators, this interface returns `[chunk.key]`
- For the shuffle operator, this interface returns `[(chunk.key, reducer_idx1), ...... , (chunk.key, reducer_idxn)]
  - For shuffle operators such as DataFrameGroupByOperand, the number of output shuffle blocks is fixed, and each mapper outputs `N` blocks, so all data keys`[(chunk.key, reducer_idx1), ...... , (chunk.key, reducer_idxn)]`
  - For shuffle operators such as `TensorReshape`, each mapper may not have the outputs to all reducers, and the shuffle data keys it outputs is determined at runtime. For this type of operator, we need to return all data keys corresponding to all reducers`[(chunk.key, reducer_idx1), ...... , (chunk.key, reducer_idxn)]`; then when reducer iter_mapper_data, skip some partitions if there is no data for key. 
  ![image](https://user-images.githubusercontent.com/12445254/169472129-f78828d7-5855-4386-837c-19208f848dcc.png)


This solution has some drawbacks:
- The metadata overhead would be very large, which will lead to supervisor OOM and task lineage evict. Assuming that we want to support `100T` data shuffle, the data size of each input chunk is 1GB, and the number of mappers and reducers is `100,000` respectively, then the data keys of each subtask will occupy about 4M memory, as shown in Figure 1, plus `FetchShuffle` stores source_keys/source_idxes/source_mappers, the data will expand by 3 times, that is, a single subtask will occupy `16M` memory. If the supervisor memory limit is 16G, this will cause the ray backend OOM when submitting 1000 subtasks.
![image](https://user-images.githubusercontent.com/12445254/169472798-535b8229-8d82-473b-82a4-26136270a9c7.png)
- Introduce shuffle operands implement burden because `get_data_keys` needs to be impelmented. 

### Building shuffle DAG by reducer index
Acutally, Mars' ChunkGraph has information about the upstream and downstream chunk dependencies. After converting to Ray DAG, this part of the information still exists. The reason for relying on data keys is that Mars replaced `ShuffleProxy` with `FetchShuffle`. The implementation of `Fetch/FetchShuffle` is an abstraction of P2P, which is actually inconsistent with the semantics of Ray DAG. Therefore, when using Ray Backend, a new set of Shuffle operators can be introduced to remove FetchShuffle and rely on Ray to automatically resolve subtask input arguments.

#### Operand Changes
- Add `n_reducers` and `reducer_ordinal` to all `MapReduceOperand` operands
- Introduce a new group of `FetchShuffle` operands which only records `n_mappers` and `n_reducers` to reduce meta overhead:
```python
class FetchShuffleType(Enum):
    FETCH_SHUFFLE_BY_INDEX = 0
    FETCH_SHUFFLE_BY_DATA_KEY = 1

class FetchShuffleByIndex(Operand):
    _op_type_ = opcodes.FETCH_SHUFFLE_BY_INDEX
    n_mappers = Int32Field("n_mappers")
    n_reducers = Int32Field("n_reducers")

class FetchShuffle(Operand):
    _op_type_ = opcodes.FETCH_SHUFFLE_BY_DATA_KEY
    source_keys = ListField("source_keys", FieldTypes.string)
    source_idxes = ListField("source_idxes", FieldTypes.tuple(FieldTypes.uint64))
    source_mappers = ListField("source_mappers", FieldTypes.uint16)
```
- Make shuffle operands mapper produce output with deterministic number and order.
    - `TensorReshape` and `TensorBinCount` mapper needs change. 
- _bagging.py uses `source_idxes`, which alsoe needs to be refactored.

#### Graph Changes
Add an option to replace `ShuffleProxy` to `FetchShuffleByIndex` when building subtask graph

#### Using a ShuffleManager to manage Ray shuffle execution
```python
class ShuffleManager:
    def __init__(self, subtask_graph):
        self.subtask_graph = subtask_graph
        # Build mapper index
        # subtask: (shuffle_index, mapper_index)
        # Build reducer index
        # subtask: (shuffle_index, reducer_ordinal)
    
    def has_shuffle(self):
        return self.num_shuffles > 0

    def add_mapper_output_refs(
        self, subtask, output_object_refs: List["ray.ObjectRef"]
    ):
        pass

    def get_reducer_input_refs(self, subtask) -> List["ray.ObjectRef"]:
        pass

    def get_n_reducers(self, subtask):
        pass
```

#### Build Shuffle DAG code
Load reducer mapper inputs:
```
    # shuffle meta won't be recorded in meta service, query it from shuffle manager.
    input_object_refs = shuffle_manager.get_reducer_input_refs(subtask)
    ray_executor.options(
            num_returns=output_count, max_retries=max_retries
        ).remote(
            subtask.task_id,
            subtask.subtask_id,
            serialize(subtask_chunk_graph),
            subtask_output_meta_keys,
            *input_object_refs,
        )
```
The final graph in ray will be:
![image](https://user-images.githubusercontent.com/12445254/169473353-98aca52e-524f-459d-92da-6b11d7bd207f.png)


# Follow-on Work
- Push-based Shuffle using Ray Future DAG
- Mitigate Straggler tasks
- Dynamically coalescing shuffle partitions
- Dynamically splitting shuffle partitions