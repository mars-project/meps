# MEP 0002: Simplify Data Structure

<!-- TOC -->

- [MEP 0002: Simplify Data Structure](#mep-0002-simplify-data-structure)
    - [General Motivation](#general-motivation)
    - [Design and Architecture](#design-and-architecture)
        - [Chunk and ChunkData merged to Chunk](#chunk-and-chunkdata-merged-to-chunk)
        - [Modification of Operand](#modification-of-operand)
        - [Subtask Fields](#subtask-fields)
        - [Other Optimization](#other-optimization)
    - [Follow-on Work and Plan](#follow-on-work-and-plan)

<!-- /TOC -->

## General Motivation

The main problem of Mars right now is performance. We did a lot of experiments and tests comparing with Dask on a single node and multiple nodes. And the results are shown in the tables below:

| **Tasks/Subtasks** | **Dask** | **Mars** | **Mars on Ray** | **Mars on Ray DAG** |
| --- | --- | --- | --- | --- |
| 2_000 | 904.98 | 82.69 | 45.97 | 356.29 |
| 20_000 | 898.27 | 86.20 | 45.54 | 342.09 |
| 40_000 | 846.92 | 85.29 | 45.73 | 343.48 |
| 200_000 | 827.36 | 78.61 | 44.59 | 326.48 |

Table 1: Tps of Dasks and Mars runing on a Node

| **Tasks/Subtasks** | **Dask** | **Mars** | **Mars on Ray** | **Mars on Ray DAG** |
| --- | --- | --- | --- | --- |
| 2_000 | 1456.31 | 149.51 | 94.73 | 468.02 |
| 20_000 | 1501.13 | 130.43 | 92.92 | 439.88 |
| 40_000 | 1366.09 | 128.68 | 86.69 | 438.97 |
| 200_000 | 1220.73 | 129.85 | 96.65 | 369.47 |

Table 2: Tps of Dasks and Mars runing on three Nodes
> Tasks/Subtasks: the number of tasks of Dask job, the number of subtasks of Mars job.

We can see that tps of Mars is much smaller than Dask regardless of a single node or distributed scenarios. And we draw three conclusions after detailed analysis:

- Graph generation is much slower than Dask. There is 3 different graphs in Mars: `TileableGraph`, `ChunkGraph` and `SubtaskGraph` while Dask has only a `HighLevelGraph`. And `ChunkGraph` and `SubtaskGraph` generation are very slow, especailly `SubtaskGraph` generation.
- Mars serialization and deserialization takes longer than Dask, which is related to two issues: data structure and serialization and deserialization method.
- Mars has much more rpc than Dask, because Dask does a batch send for all rpcs, even if those rpcs are of different types.

The first two issues are related to data structure. There are several problems with complex data structures:

- The main data structure of the graph is `ChunkData`. If it takes a long time to create new chunk data and there are many chunk data, then graph generation will be very time-consuming.
- Serialization and deserialization data is larger and takes longer time.

So, this proposal we focus on addressing the first two issues by simplifying the data structure. And we'll optimize the serialization later.

## Design and Architecture

We mainly makes the following optimizations in simplifying the data structure.

### Chunk and ChunkData merged to Chunk

In order to address the main problem of graph generation, we profiled the supervisor and got the following flame graph.

![mep-0002_01](https://user-images.githubusercontent.com/5388750/231441694-82cba4a4-ecca-4949-8445-363fd51c9673.png)

The graph is the tile process of ChunkGraph generation. We can see from the figure that the chunk creatings took too long. And the reason is that ChunkData has too many fields and its inheritance relationship is complicated.
The current data structure of `Chunk` and `ChunkData`are:

![mep-0002_02](https://user-images.githubusercontent.com/5388750/231441707-1a0e2cad-bb85-4b01-bedf-05f768fe27ab.png)

![mep-0002_03](https://user-images.githubusercontent.com/5388750/231441708-5bc6c52f-cfb5-4601-a4b1-e9fe78a68313.png)

1. Remove `type_name` and we can use `self.__class__.__name__` instead when necessary.

    Currently the main place of use is `DataFrameData`and `CategoricalData` like:

    ```python
    class DataFrameData(_BatchedFetcher, BaseDataFrameData):
        type_name = "DataFrame"

        def _to_str(self, representation=False):
            if is_build_mode() or len(self._executed_sessions) == 0:
                # in build mode, or not executed, just return representation
                if representation:
                    return (
                        f"{self.type_name} <op={type(self._op).__name__}, key={self.key}>"
                    )
                else:
                    return f"{self.type_name}(op={type(self._op).__name__})"
            else:
                ...
    ```

    ```python
    class CategoricalData(HasShapeTileableData, _ToPandasMixin):
        ...
        
        def _to_str(self, representation=False):
            if is_build_mode() or len(self._executed_sessions) == 0:
                # in build mode, or not executed, just return representation
                if representation:
                    return f"{self.type_name} <op={type(self.op).__name__}, key={self.key}>"
                else:
                    return f"{self.type_name}(op={type(self.op).__name__})"
            else:
                data = self.fetch(session=self._executed_sessions[-1])
                return repr(data) if repr(data) else str(data)
    ```

2. Remove `__allow_data_type__` which is used to check the valid of chunk data. If we remove `ChunkData`, it doesn't need to exist.
3. Remove `_id`and keep only `_key`.
4. Express operand with op id and args to replace operand instance.
5. Remove `Entity`and `EntityData` to reduce class inheritance hierarchy.

Finally we will get the Chunk:

```python
class Chunk(Serializable):
    __slots__
    
    _key: str
    _op_id: str
    _op_args: Tuple
    _inputs: Tuple
    _outputs: Tuple
    
    _index: Tuple
    _is_broadcaster: bool
    _extra_params: Dict
```

**1-3** is to reduce fields like the `_id`. We did a comparison test.
Firstly we define a simple `TensorChunkData`:

```python
class TensorChunkData(Serializable):
    _shape = TupleField("shape", FieldTypes.int64)
    _order = ReferenceField("order", TensorOrder)
    _index = TupleField("index", FieldTypes.uint32)
    _extra_params = DictField("extra_params", key_type=FieldTypes.string)
    _key = StringField("key", default=None)
    _id = StringField("id")

    def __init__(self, op=None, index=None, shape=None, dtype=None, order=None, **kw):
        args = ()
        kwargs = {
            "_shape": shape,
            "_order": order,
            "_op": op,
            "_index": index,
            "_extra_params": {"_i": kw["_i"]},
            "_key": kw["_key"],
            "_dtype": kw.get("_dtype"),
            "_id": str(id(self)),
        }
        super().__init__(*args, **kwargs)
```

![mep-0002_04](https://user-images.githubusercontent.com/5388750/231441711-9cd42d03-1c9c-4089-8b1c-ee70699d0965.png)

And we can see from the graph the cost of constructing the `TensorChunkData`is about 2.81e-06s.
Secondly, we remove the `_id` field and then do the same constucting.

![mep-0002_05](https://user-images.githubusercontent.com/5388750/231441715-6b26adae-9dcf-4fbe-86c9-f727b06cf674.png)

The cost becames 2.26e-06s, which is reduced **19.6%**.

**4** is to change the operand instance to parameters. And also we did a comparison.

```python
from mars.serialization.core import serialize

op = TensorRandomSample(seed=1001, size=100, gpu=False, dtype=np.dtype("f8"))
bs1 = pickle.dumps(serialize(op))

args = ("op_001", (1001, 100, False, np.dtype("f8")))
bs2 = pickle.dumps(serialize(args))
```

The data size is:

![mep-0002_06](https://user-images.githubusercontent.com/5388750/231441718-4f99205d-4d88-42fc-9d24-5a659a47a1b9.png)

The serializtion cost is:

![mep-0002_07](https://user-images.githubusercontent.com/5388750/231441724-001346e1-5d03-4f5a-9903-c591f828e927.png)

We can see that data size is reduced **71.5%**, and cost reduced **63.8%**.
**5** is to reduce the inheritance level of the class and the test result is as follow.

Firstly, we define a `TensorChunkData` which extends `EntityData`:

```python
class EntityData(Serializable):
    _key = StringField("key", default=None)
    _id = StringField("id")
    _extra_params = DictField("extra_params", key_type=FieldTypes.string)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TensorChunkData(EntityData):
    _shape = TupleField("shape", FieldTypes.int64)
    _order = ReferenceField("order", TensorOrder)
    _index = TupleField("index", FieldTypes.uint32)

    def __init__(self, op=None, index=None, shape=None, dtype=None, order=None, **kw):
        args = ()
        kwargs = {
            "_shape": shape,
            "_order": order,
            "_op": op,
            "_index": index,
            "_extra_params": {"_i": kw["_i"]},
            "_key": kw["_key"],
            "_dtype": kw.get("_dtype"),
            "_id": str(id(self)),
        }
        super().__init__(*args, **kwargs)

```

We constructed a `TensorChunkData`:

![mep-0002_08](https://user-images.githubusercontent.com/5388750/231441726-80de2790-29c9-4666-b6de-3315ad13ebac.png)

The cost is about 3.55e-06s.

Secondly, we define a `TensorChunkData` which extends `Serializable`directly.

```python
class TensorChunkData(Serializable):
    _shape = TupleField("shape", FieldTypes.int64)
    _order = ReferenceField("order", TensorOrder)
    _index = TupleField("index", FieldTypes.uint32)
    _extra_params = DictField("extra_params", key_type=FieldTypes.string)
    _key = StringField("key", default=None)
    _id = StringField("id")

    def __init__(self, op=None, index=None, shape=None, dtype=None, order=None, **kw):
        args = ()
        kwargs = {
            "_shape": shape,
            "_order": order,
            "_op": op,
            "_index": index,
            "_extra_params": {"_i": kw["_i"]},
            "_key": kw["_key"],
            "_dtype": kw.get("_dtype"),
            "_id": str(id(self)),
        }
        super().__init__(*args, **kwargs)
```

We also consturcted a `TensorChunkData`:

![mep-0002_09](https://user-images.githubusercontent.com/5388750/231441728-af08d374-d327-45c0-803d-c611f4dda7ad.png)

The cost is about 2.81e-06s, which is reduced **20.8%**.

There is another point to reduce time-consuming effect of **5**.

Firstly, we define a `TensorChunk`:

```python
class TensorChunk(Serializable):
    type_name = "Tensor"
    _data = ReferenceField("data", EntityData)

    def __init__(self, data, **kwargs):
        super().__init__(_data=data, **kwargs)
```

We constructed a `TensorChunk`:

![mep-0002_10](https://user-images.githubusercontent.com/5388750/231441729-22081b47-241a-4944-a283-70ae34795194.png)

The cost is about 8.28e-07s which accounts for **29.5%** of constructing a `TensorChunkData`. We can save this time consuming if we merge `Chunk`and `ChunkData` into `Chunk`.

### Modification of Operand

We need to generated a `op_id`field for every Operand, and maintain a mapping from `op_id`to Operand like follows so as to find the corresponding Operand and construct the Operand instance.
For convenience, we take the following steps to generate `op_id` and mapping:

- `op_id = hash("Operand's full path")`
- The key of mapping is `op_id`, and value is initialized as operand's full path like `mars.tensor.random.core.TensorRandomOperand`.
- When constructing op, first check the type of value in mapping, if it is str, load the corresponding operand class, and update value to operand class, if it is class, then use it directly.
- Construct the mapping when building pymars wheel.

> The hash function is not the builtin `hash()`. Because Python 3.4+ switched the hash function to SipHash for security (to avoid collision attack), the same string has different hash values in different Python processes. We will use murmurhash here.

And the mapping is like:

```python
OPERAND_MAPPING = {
    574610788: 'mars.tensor.random.core.TensorRandomOperand' or TensorRandomOperand,
    1112862296: 'mars.core.operand.shuffle.MapReduceOperand',
    710700605: 'mars.core.operand.fetch.FetchShuffle',
	...
}
```

### Subtask Fields

1. Remove `subtask_name` and only `TaskProcessorActor.get_tileable_subtasks`use it currently.
2. The `task_id` could be directly generated by a global sequencer, the `stage_id` is generated by the task-level sequencer, and the `subtask_id` is generated by the stage-level sequencer. Finally, we can compose `subtask_id` like: 

    ```python
    def gen_subtask_id(task_id, stage_id, subtask_id):
        return task_id.to_bytes(4, 'little') + 
            stage_id.to_bytes(2, 'little') + 
            subtask_id.to_bytes(6, 'little')

    subtask_id = gen_subtask_id(task_id, stage_id, subtask_id)

    # i.e.
    # task_id(4 bytes) | stage_id(2 bytes) | subtask_id(6 bytes)
    ```

    For example:

    ```python
    # if
    task_id = 1
    stage_id  = 1
    subtask_id = 1
    # then
    # subtask_id = b'\x01\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00'
    # and subtask_id.hex() is `010000000100010000000000`
    ```

    We cat get `task_id` and `stage_id` as follows:

    ```python
    task_id = subtask_id[0::4]
    subtask_id = subtask_id[4::6]
    ```

    We compared the new subtask id generation method with the old method and the results are as follows:

    ![mep-0002_11](https://user-images.githubusercontent.com/5388750/231441731-27b3b28b-f0cc-4dd2-bb7d-a011422c18c6.png)

    The time consumption is reduced to **2.9%** of the original. And the bytes are reduced **from 24 to 12**.

And finally the fields of Subtask are:

```python
class Subtask(Serializable):
    subtask_id: str
    chunk_graph: ChunkGraph
    
    expect_bands: List
    virtual: bool
    retryable: bool
    priority: Tuple
    rerun_time: int
    extra_config: Dict
    update_meta_chunks: List
    
    logic_key: str
    logic_index: int
    logic_parallelism: int
    
    bands_specified: bool
    required_resource: Resource
    execution_time_mills: int
    submitted: bool
```

Later we will move some fields of `Subtask` to `SubtaskScheduleInfo`, and `Subtask` only keep the necessary fields at runtime, like:

```python
class Subtask(Serializable):
    subtask_id: str
    chunk_graph: ChunkGraph
```

### Other Optimization

1. Simplify the key generation of `Chunk` by using `op key + chunk index`. We only need to generate the key once instead of generating the same number of keys as the number of chunks.
2. We construct the same Chunk of ChunkGraph when generating a SubtaskGraph.

    We did a comparison: one creating new chunk and the other does not. The result is that the cost is reduced from 43.4s to 14.6s which is about **66.4%** reduced. But there is one point that `FetchChunk` needs to be created.

3. We generate logic_key once for all operands or subtasks, in fact, it is possible to generate only once for operand and subtask of the same type.

## Follow-on Work and Plan

1. Optimize the chunks creation in generating `SubtaskGraph`.
2. Remove `_id`and only keep the `_key` of `TileableData`, `ChunkData`and `Operand`.
3. Merge `Chunk`and `ChunkData`into `Chunk` and simplify the `_key` generation of `Chunk`.
4. Change operand instance to operand parameters in `ChunkData` and modify the operand.
5. Simplify `Subtask`
6. Optimize the `logic_key` generation.
