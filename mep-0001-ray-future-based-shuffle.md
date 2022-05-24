# Motivation
Shuffle is the core operation for large-scale data processing. The data is partitioned according to the partition key in each record through the ALL-TO-ALL connection, and sent from all mappers to all reducers. Due to the nature of its ALL-TO_ALL connections, shuffle often becomes the bottleneck of the performance, stability and scalability of the entire system. This MEP introduces a pull-based shuffle implementation using Ray future which serves following goals:
- Improve Mars-on-Ray shuffle scalability by reducing shuffle meta
- Improve scheduling performance by using Ray C++ scheduler
- Support shuffle failover by lineage reconstruction
- Setup the foundation for push-based shuffle using Ray future
 
## Design and Architecture
When the Mars subtask has upstream and downstream dependencies, the downstream subtask's dependence on the output of the upstream subtask is determined by the data key. The input data required for subtask execution is a context dict object from data key to data.
 
For non-shuffle operators, the data key is the chunk key; but for shuffle operators, the data key is a tuple of (chunk key, reducer index).
 
Therefore, in order to build a Shuffle DAG based on Ray future, either all the data keys output by each mapper subtask are known in advance, or the data keys can be mocked when executing subtask.
 
### Building shuffle DAG by reducer index
Actually, Mars' ChunkGraph has information about the upstream and downstream chunk dependencies. After converting to Ray DAG, this part of the information still exists. The reason for relying on data keys is that Mars replaced `ShuffleProxy` with `FetchShuffle`. The implementation of `Fetch` and `FetchShuffle` is an abstraction of P2P, which is inconsistent with the semantics of Ray DAG actually. Therefore, when using Ray Backend, a new set of Shuffle operators can be introduced to replace `FetchShuffle` and rely on Ray to automatically resolve subtask dependencies.
 
#### Operand Changes
- Add `n_reducers` and `reducer_ordinal` to all `MapReduceOperand` operands
- Introduce a new group of `FetchShuffle` operands which only records `n_mappers` and `n_reducers` to reduce meta overhead:
```python
class FetchShuffleType(Enum):
   FETCH_SHUFFLE_BY_INDEX = 0
   FETCH_SHUFFLE_BY_DATA_KEY = 1
 
class FetchShuffleByIndex(Operand):
   _op_type_ = opcodes.FETCH_SHUFFLE_BY_INDEX
   # number of all upstream mappers
   n_mappers = Int32Field("n_mappers")
   # number of all downstream reducers
   n_reducers = Int32Field("n_reducers")
 
class FetchShuffle(Operand):
   _op_type_ = opcodes.FETCH_SHUFFLE_BY_DATA_KEY
   source_keys = ListField("source_keys", FieldTypes.string)
   source_idxes = ListField("source_idxes", FieldTypes.tuple(FieldTypes.uint64))
   source_mappers = ListField("source_mappers", FieldTypes.uint16)
```
- Make shuffle operands mapper produce output with deterministic number and order.
   - `TensorReshape` and `TensorBinCount` mapper needs change.
- _bagging.py uses `source_idxes`, which also needs to be refactored.
 
#### Graph Changes
Add an option to replace `ShuffleProxy` to `FetchShuffleByIndex` when building subtask graph.
 
#### Using a ShuffleManager to manage Ray shuffle execution
`ShuffleManager` is defined in `mars/services/task/execution/ray/shuffle.py` and only used by Ray executor. When `TaskProcessorActor` invokes `RayTaskExecutor.execute_subtask_graph` to execute a subtask graph, The `RayTaskExecutor` will create a `ShuffleManager` to manage shuffle execution.
 
The `ShuffleManager` will do following things:
- Build mapper index which is a dict from mapper subtask to (`shuffle_index`, `mapper_index`). A subtask graph may have multiple groups of shuffles, `shuffle_index` indicates a mapper subtask belongs to which shuffle. `mapper_index` indicates is ordinal in all mapper subtasks of current shuffle.
- Build reducer index which is a dict from reducer subtask to (`shuffle_index`, `reducer_ordinal`). A subtask graph may have multiple groups of shuffles, `shuffle_index` indicates a reducer subtask belongs to which shuffle. `reducer_ordinal` indicates ordinal in all reducer subtasks of current shuffle. If some reducers are missing, `ShuffleManager` will fill `None` for those reducers index.
- Recording mapper output object refs for all mapper subtasks, which will be used by later reducers to get all mapper inputs based on reducer ordinal.
- Return `n_reducers` for a shuffle when passing a mapper/reducer subtask.
- In the future, it will manage push-based shuffle scheduling and execution.
 
The code skeleton is as follows:
 
```python
class ShuffleManager:
    """Manage shuffle execution for ray by resolve dependencies between mappers outputs and reducers inputs based on
    mapper and reducer index.
    """

    def __init__(self, subtask_graph):
        self.subtask_graph = subtask_graph
        # Build mapper index
        # subtask: (shuffle_index, mapper_index)
        # Build reducer index
        # subtask: (shuffle_index, reducer_ordinal)

    def has_shuffle(self) -> bool:
        """
        Returns
        -------
        bool
            Whether current subtask graph has shuffle to execute
        """
        return self.num_shuffles > 0

    def add_mapper_output_refs(self, subtask, output_object_refs: List["ray.ObjectRef"]):
        """
        Record mapper output ObjectRefs which will be used by reducers later.

        Parameters
        ----------
        subtask
        output_object_refs : List["ray.ObjectRef"]
            Mapper output ObjectRefs.
        """
        pass

    def get_reducer_input_refs(self, subtask: Subtask) -> List["ray.ObjectRef"]:
        """
        Get the reducer inputs ObjectRefs output by mappers.

        Parameters
        ----------
        subtask : Subtask
            A reducer subtask.

        Returns
        -------
        input_refs : List["ray.ObjectRef"]
            The reducer inputs ObjectRefs output by mappers.
        """
        pass

    def get_n_reducers(self, subtask: Subtask):
        """
        Get the number of shuffle blocks that a mapper operand outputs,
        which is also the number of the reducers when tiling shuffle operands.
        Note that might be greater than actual number of the reducers in the subtask graph,
        because some reducers may not be added to chunk graph.

        Parameters
        ----------
        subtask : Subtask
            A mapper or reducer subtask.

        Returns
        -------
        n_reducers : int
            The number of shuffle blocks that a mapper operand outputs.
        """
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
The final graph in Ray will be:
![image](https://user-images.githubusercontent.com/12445254/169473353-98aca52e-524f-459d-92da-6b11d7bd207f.png)
 
# Alternative Solution
### Build shuffle DAG using data key
If data keys are used, all the data keys output by each mapper subtask must be known in advance, then pack those data keys and data into a dict and pass it to the subtask when the subtask is submitted.
 
A new `get_data_keys()` interface needs to be added to all shuffle operators to obtain all the data keys of the subtask in advance:
- For non-shuffle operators, this interface returns `[chunk.key]`
- For the shuffle operator, this interface returns `[(chunk.key, reducer_idx1), ...... , (chunk.key, reducer_idxn)]
 - For shuffle operators such as DataFrameGroupByOperand, the number of output shuffle blocks is fixed, and each mapper outputs `N` blocks, so all data keys`[(chunk.key, reducer_idx1), ...... , (chunk.key, reducer_idxn)]`
 - For shuffle operators such as `TensorReshape`, each mapper may not have the outputs to all reducers, and the shuffle data keys it outputs is determined at runtime. For this type of operator, we need to return all data keys corresponding to all reducers`[(chunk.key, reducer_idx1), ...... , (chunk.key, reducer_idxn)]`; then when reducer iter_mapper_data, skip some partitions if there is no data for key.
 ![image](https://user-images.githubusercontent.com/12445254/169472129-f78828d7-5855-4386-837c-19208f848dcc.png)
 
 
This solution has some drawbacks:
- The metadata overhead would be very large, which will lead to supervisor OOM and task lineage evict. Assuming that we want to support `100T` data shuffle, the data size of each input chunk is 1GB, and the number of mappers and reducers is `100,000` respectively, then the data keys of each subtask will occupy about 4M memory, as shown in Figure 1, plus `FetchShuffle` stores source_keys/source_idxes/source_mappers, the data will expand by 3 times, that is, a single subtask will occupy `16M` memory. If the supervisor memory limit is 16G, this will cause the Ray backend OOM when submitting 1000 subtasks.
![image](https://user-images.githubusercontent.com/12445254/169472798-535b8229-8d82-473b-82a4-26136270a9c7.png)
- Introduce burden to implement shuffle operands because `get_data_keys` needs to be implemented.
 
# Follow-on Work
- Push-based Shuffle using Ray Future DAG
- Mitigate Straggler tasks
- Dynamically coalescing shuffle partitions
- Dynamically splitting shuffle partitions