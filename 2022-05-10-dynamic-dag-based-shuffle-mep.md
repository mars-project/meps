# Dynamic DAG based Large-Scale Shuffle for Mars
## Motivation
Shuffle is the core operation for large-scale data processing. The data is partitioned according to the partition key in each record through the ALL-TO-ALL connection, and sent from all mappers to all reducers. Due to the nature of its ALL-TO_ALL connections, shuffle often becomes the bottleneck of the performance, stability and scalability of the entire system. This MEP introduces a dynamic dag based high-performance, adaptive shuffle framework for Mars that supports petabyte-level data processing.

## Proposal - Dynamic DAG based Shuffle
The existing Shuffle systems are mainly divided into two types: Pull and Push. Pull based Shuffle can increase the efficiency of IO through pre-shuffle merge. Push-based Shuffle can increase IO and scheduling pipeline by pushing data to reducer nodes in advance. Our Dynamic DAG solution is based on Push-Shuffle.
![image](https://user-images.githubusercontent.com/12445254/169471962-521b0ec1-b65e-4a5a-b4d8-021b02a8ee4c.png)


Dynamic DAG based shuffle consists of:
- Pull-based Shuffle using Ray Future DAG
- Push-based Shuffle using Ray Future DAG
- Mitigate Straggler tasks
- Dynamically coalescing shuffle partitions
- Dynamically splitting shuffle partitions
- Support failover by lineage recontruction

## Pull-based Shuffle for Ray DAG
When the Mars subtask has upstream and downstream dependencies, the downstream subtask's dependence on the output of the upstream subtask is determined by the data key. The input data required for subtask execution is a context dict object from data key to data. For non-shuffle operators, the data key is the chunk key; but for shuffle operators, the data key is a tuple of (chunk key, reducer index). Therefore, in order to build a Shuffle DAG based on Ray, either all the data keys output by each mapper subtask are known in advance, or the data keys are removed.

### Data Key based Shuffle DAG
If data keys are used, all the data keys output by each mapper subtask must be known in advance, then pack those data keys and data into a dict and pass it to the subtask when the subtask is executed. `num_returns` can be determined according to the number of data keys.
In order to obtain all the data keys of the subtask in advance, a new `get_data_keys()` interface needs to be added to all operators:
- For non-shuffle operators, this interface returns `[chunk.key]`
- For the shuffle operator, this interface returns `[(chunk.key, reducer_idx1), ...... , (chunk.key, reducer_idxn)]
  - For shuffle operators such as DataFrameGroupByOperand, the number of output shuffle blocks is fixed, and each mapper outputs `N` blocks, so all data keys`[(chunk.key, reducer_idx1), ...... , (chunk.key, reducer_idxn)]`
  - For shuffle operators such as `TensorReshape`, each mapper may not have the outputs to all reducers, and the shuffle data keys it outputs is determined at runtime. For this type of operator, we can directly return all data keys corresponding to all reducers`[(chunk.key, reducer_idx1), ...... , (chunk.key, reducer_idxn)]`; then when reducer iter_mapper_data, skip some partitions if there is no data for key. 
  ![image](https://user-images.githubusercontent.com/12445254/169472129-f78828d7-5855-4386-837c-19208f848dcc.png)

### Ray Future based Shuffle DAG
The main issue of data keys based solution is that the metadata overhead would be very large, which will lead to supervisor OOM and task lineage evict.
Assuming that we want to support `100T` data shuffle, the data size of each input chunk is 1GB, and the number of mappers and reducers is `100,000` respectively, then the data keys of each subtask will occupy about 4M memory, as shown in Figure 1, plus `FetchShuffle` stores source_keys/source_idxes/source_mappers, the data will expand by 3 times, that is, a single subtask will occupy `16M` memory. If the supervisor memory limit is 16G, this will cause the ray backend OOM when submitting 1000 subtasks.
![image](https://user-images.githubusercontent.com/12445254/169472798-535b8229-8d82-473b-82a4-26136270a9c7.png)

Acutally, Mars' ChunkGraph has information about the upstream and downstream chunk dependencies. After converting to Ray DAG, this part of the information still exists. The reason for relying on data keys is that Mars replaced `ShuffleProxy` with `FetchShuffle`. The implementation of `Fetch/FetchShuffle` is an abstraction of P2P, which is actually inconsistent with the semantics of Ray DAG. Therefore, when using Ray Backend, a new set of Shuffle operators can be introduced to remove FetchShuffle and rely on Ray to automatically resolve subtask input arguments.
![image](https://user-images.githubusercontent.com/12445254/169473353-98aca52e-524f-459d-92da-6b11d7bd207f.png)

Shuffle DAG code:
```
@ray.remote(num_returns=R)
def map(m):
    data = load_partition(m)
    return [data[s:e] for s, e in sort_and_partition(data, R)]
@ray.remote
def reduce(blocks):
    return merge_sorted(blocks)
    def shuffle(M, R):
map_out = [map.remote(m) for m in range(M)]
return [reduce.remote(map_out[:,r]) for r in range(R)]
```

potential issues:
- _bagging.py uses `source_idxes`, which needs to be refactored.

## Implement Push-based Shuffle to improve scale and efficiency
Although Pull-based Ray DAG Shuffle resolved the single-threaded bottleneck of Mars python scheduling, there are still multiple issues that cause shuffle to be the bottleneck of the entire system:
- Scheduling cannot be pipelined: the driver needs to wait for all mappers to complete before scheduling the reducer. This synchronous barrier will severely impact pipeline scheduling and execution, resulting in the entire shuffle execution time too long.
- The IO efficiency is too low: In Pull-based Shuffle, all reducer shuffle fetch requests will arrive at the object store of the mapper node in random order, so the object store will also randomly access the data, if the data is spilled to the disk, these random reads will seriously affect the disk throughput and increase the shuffle wait time.
- Affects the shuffle efficiency of other jobs: Because the object store is a service shared by all jobs, a job with low shuffle efficiency will affect the shuffle efficiency of other jobs. When a job generates a large number of shuffle blocks that spill to disk, the shuffle efficiency of other jobs will also be affected because the object store requires a lot of random access to the disk.
- The `M*N` metadata overhead is too high and OOM cannot support larger-scale shuffle tasks. For a 512GB terasort, Ray-Simple cannot run with a partition size of 100M, because each shuffle block is only 26M in this case, and the total metadata overhead will cause driver OOM. That is, shuffle tasks with more than 5120 partitions cannot be run.

These issues can be resolved by future-based push shuffle proposed by Ray Exoshuffle. Push-based Shuffle pushes the shuffle block to the reducer side in advance, an merges multiple blocks belonging to the same reducer. The topology is shown in Figures c and d below. In this way, Shuffle data-scale, IO efficiency, scheduling pipeline, and final reducer locality can be improved.
![image](https://user-images.githubusercontent.com/12445254/169474732-062cd91c-95ef-4578-b2bc-08f0b23c002b.png)

### Ray dynamic DAG based Push Shuffle
The push-based shuffle based on Ray DAG consists of the following steps:
1. 
In order to specify the affinity between ray tasks, first use placement group to obtain resources, schedule all task graphs in pg, and specify the affinity of shuffle tasks through pg bundle.
  a. All tasks must be scheduled in pg, not just using pg to achieve the affinity of mapper task and merge task. Otherwise, since the upstream input node is not in the applied pg, the mapper task will not be able to be scheduled to the upstream input node.
2. For a single SubtaskGraph, divide all mapper and reducer subtasks belonging to the same shuffle.
  a. Since the previous Simple Shuffle implementation did not build Fetch Chunk, the graph structure has not changed. You can directly find ShuffleProxy, and then its upstream subtask is the same Shuffle's mapper subtask, and its downstream subtask is the same Shuffle's reducer subtask
3. First assign a node (ie pg bundle) to each reducer. For simplicity, the strategy can be RoundRobin, and then start submitting multiple rounds of mapper and merge.
4. First submit a set of mapper subtasks, and specify the number of shuffle blocks returned by num_returns.
5. Wait for the execution of the previous round of merge tasks to complete, and then submit the merge task for this round of mappers.
  a. For each reducer, submit multiple merge tasks according to the merge factor, and specify the pg bundle of the merge task to ensure that the merge task is scheduled on the reducer node.
  b. Judge the merge straggler, cancel this part of the merge, and directly use the output of the mapper
6. Loop through steps 4 and 5 until all mappers are executed
7. Submit the reducer task according to the locality specified in advance for the reducer, specifying the merge output as a parameter
  a. When submitting the reducer, sort the merger output and mapper output parameters in the order of dependencies of the subtask graph, and then pass it to the reducer task to ensure that the reducer gets all the parameters in the correct order of shuffle blocks


pseudo-code:
```python
# TODO: Code for speculative execution, adaptive merge factor, adaptive P value

# The core logic is to submit the next round of mappers for CPU/Memory-intensive computation first, 
# then wait for the completion of the network-intensive merge computation submitted in the previous round, 
# then submit the next round computation. In this way, it can be ensured that only one round of mappers and reducers is executed at the same time, 
# so that the CPU, memory, disk, and network of the system have high utilization rate.。
merge_out = []
out = [] # output from one round of merge tasks
for round in range(M / P):
    map_out = [map.remote(m) for m in range(P)]
    ray.wait(out)
    out = ... # can be either Riffle- or Magnet-style merge
    merge_out.extend(out)
    # delete mapper output to make shuffle block gc more timely.
    # With proper `P` value, all mapper outputs will be in memory, thus aovid disk-write
    del map_out

reducer_out = [reduce.remote(merge_out[:,r]) for r in range(R)]
```

The executing process is as follows:
![image](https://user-images.githubusercontent.com/12445254/169737023-169e47ae-f1c7-44a0-b446-84bd9fde31d5.png)


The key to improving the efficiency of push-based shuffle is:
- The choice of merge factor, which determines whether mapper and merge execution can be pipelined. In Shuffle, mapper subtasks generally chain some upstream operators, which is a CPU and memory-intensive calculation. While merger only merges data, which is an IO and network-intensive task. By submitting mapper and merger alternately in multiple rounds, each component of the system can have a higher utilization rate.
  - Supports configure merge factor. Because in the pipeline execution, we need to wait for the previous round of merge execution to complete before submitting the next round of mapper, so the merge time needs to be lower than the mapper time, otherwise the mapper submission will be too late and this will negative optimization. For example, map takes twice as long as merge, so each merge task should be the input of a single partition of two maps.
  - Supports automatic selection of merge factors, that is, how many mappers correspond to a reducer. We can set a smaller merge factor at first, run a round of mapper and merge task first, then compare the execution time, and determine the merge factor. Or better yet, resample the consumed time every few rounds and recalculate the merge factor.
- In addition, due to network and node environment problems, straggler may appear in the merge, which will also cause the next round of mappers to fail to be submitted. Therefore, it is necessary to dynamically cancel some mergers that have timed out. For example, only wait for 95% of the merges to succeed, and discard the rest.
- Reduce write amplification. By limiting the number of mapper tasks that are executed at the same time, it is possible that the output generated by each mapper round does not exceed the object store memory, thereby avoid write amplification. How many mappers are runnign  at the same time can be determined according to the size of the first round of mapper output data statistics.

### Static DAG-based Push Shuffle
The Ray DAG solution can only be used by the Ray backend, and cannot be reused by the mars backend. By building a static DAG through inserting a set of mergers between the mapper and the reducer when building the graph, it will be shared by mars and ra. The subtask graph will change from Figure 1 to a graph 2:
![image](https://user-images.githubusercontent.com/12445254/169740634-f3554959-1ef2-4cc9-a77e-a113bc085c83.png)
![image](https://user-images.githubusercontent.com/12445254/169740686-b340a712-596f-4e05-b6ed-46e57868e8f2.png)

This solution only modifies the graph, there are two ways:
1. Add a set of mergers to the tile stage of the operator. This method needs to modify each Shuffle operator
2. When generating the subtask graph, find the mapper and reducer belonging to the same set of Shuffle, and then insert a set of mappers between them.
Then, schedule the merges to corresponding node for reducer, part of the mapper data will be pulled to the reducer node in advance. The implementation of static composition is relatively simple, but there are some problems:
- It is difficult to express multiple rounds of alternate execution of mapper and merge logic, that is, pipeline scheduling cannot be archived, and straggler cannot be canceled when straggler appears in mergers.
- The optimial merge factor cannot be determined.
- The scale of graph will change from M+N level to M*N/F level, and will OOM ofr big graph. For example, the number of mappers in Figure 2 above is 6, the merge factor is 2, the number of merge tasks is 9, and the entire graph increases rapidly. If the merge factor is set to a large value, the merge subtask will take more time than the mapper, which will also lead to inability to pipeline.
- The subtask graph is generated too slowly, causing the cluster to be idle for a long time. Because the graph scale increases squarely, it will take a lot of time to generate the graph, and this time can actually be used to execute some subtasks.

## Straggler Mitigation
In Ray, mapper and merger are scheduled alternately, and the reducers won't be scheduled until all mappers are executed. If the straggler occurs in mappers or mergers, it will cause the scheduling delay of later mappers and mergers, and the scheduling of final reducers will also be delayed. Therefore, Ray Shuffle needs to support handling slow mappers and merges.
Slow mapper task processing can be solved by speculative execution. When a very slow mapper subtask is found, we can duplicate it and schedule it to a different node:
```python
map_out = ...
_, timeout_tasks = ray.wait(map_out, timeout=TIMEOUT)
duplicates = []
for task in timeout_tasks:
    duplicates.append(map.remote(task.args))
for t1, t2 in zip(timeout_tasks, duplicates):
    t, _ = ray.wait([t1, t2], num_returns=1)
    map_out[t1.id] = t
```
Speculative execution for merge task will just be wasting resources, so we directly cancel the merge when it timeout, and then let reducers use the output of mappers for this merger:
```python
map_out = ...
merge_out = ...
_, timeout_tasks = ray.wait(merge_out, timeout=TIMEOUT)
for task in timeout_tasks:
    ray.cancel(task)
    merge_out[task.id] = task.args
out = [reduce.remote(merge_out[:, r]) for r in range(R)]
ray.wait(out)
```

## Dynamically coalescing shuffle partitions
In shuffle, the number of reducers affects the performance of scheduling and execution:
- If there are too few reducers, the data of each reducer will be too much, which may lead to reducer OOM;
- If there are too many reducers, a large number of reducer tasks will need to be scheduled, increasing the scheduling overhead. In addition, small objects will have higher IO overhead than large objects.
- At the same time, due to the small amount of execution time of the reducer of the small partition, the overall reducers execution are uneven, which is **not friendly to scheduling and speculative execution**.

This can be solved by shuffle partition merging: set a larger number of reducers in the tile phase, and then dynamically sample the shuffle output block statistics of mappers when the first round of mapper or multiple rounds of mapper of push-based shuffle are executed. In this way we can determine which shuffle partitions are small and can be merged together, and then when the merger and reducers are being scheduling, the partitions data of multiple reducers can be processed in a single merge and reduce task. This will bring speed up for operators such as groupby. The process is as follows:
![image](https://user-images.githubusercontent.com/12445254/169809064-380f1755-a26c-4df9-b7c5-77398952e0e3.png)


## Dynamically spliting shuffle partitions
In the actual situation, besides small partition problem, there will also be large partition problem such as data skew. Join data skew is a common scenario. If a key is skewed, the join task will become extremely slow. Also, skewed tasks will also be **unfriendly to scheduling and speculative execution**.

It is very tricky to deal with such problems manually, which need to manually collect data statistics, and perform tedious operations such as adding salt to random split the data, or processing the data batch by btach.
Data skew is essentially caused by the uneven distribution of data among partitions. Therefore, we can split the skewed partition into small sub-partitions, and then join with the entire partition corresponding to the right table to ensure the task data. The steps are as follows:
- Dynamically collect the shuffle block stats produced by each mapper to determine whether there is a skewed partition. When the output of the shuffle partition **exceeds 5 times the median of the partition size**, and the output of the partition is greater than `256M`, it is considered that data skew occurs.
- For skewed partitions, skip push-based shuffle.
- Split skewed mapper blocks into small sub-blocks. The partition data can be divided into N parts according to the target size(`max(64M, the average size of non-data skewed partitions)`). Then duplicate the entire partition of the right table to join with skewed left table. The rest of the partitions remain unchanged.
- Dynamically update the split partitions to the chunk graph and tiled tileable. Since the skewed partition is split into multiple sub-partitions, it will correspond to multiple sub-chunks, and these chunks need to be updated to the chunk graph and tiled tileable.

For example, in the following case, Table A joins Table B, where the data of partition A1 of Table A is much larger than that of other partitions. Dynamic split partition will split partition A1 into 2 sub-partitions, and let them join with partition B1 of Table B independently.
![image](https://user-images.githubusercontent.com/12445254/169822780-623edf18-d36a-4a85-b3f9-cc3f7e2a1db2.png)
Without this optimization, Shuffle would generate 4 tasks and one of them would take much longer to execute than the other. After optimization, this join will have 5 tasks, but the execution time of each task is about the same, so the whole join execution achieves better performance.
Note: Since the dynamic skew optimization is similar to the broadcast join of skewed partitions, this optimization is not suitable for scenarios where the skewed table are outer joined.

# Failover
Failover will be handled in issue https://github.com/mars-project/mars/issues/2972


# Reference
- [Exoshuffle: Large-Scale Shuffle at the Application Level](https://arxiv.org/pdf/2203.05072.pdf)
- [SPIP: Support push-based shuffle to improve shuffle efficiency](https://issues.apache.org/jira/browse/SPARK-30602#)
- [Magnet: A scalable and performant shuffle architecture for Apache Spark](https://engineering.linkedin.com/blog/2020/introducing-magnet)
- [Spark Adaptive Query Execution](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution) 
- [Adaptive Query Execution: Speeding Up Spark SQL at Runtime](https://databricks.com/blog/2020/05/29/adaptive-query-execution-speeding-up-spark-sql-at-runtime.html)
- [Node affinity scheduling strategy](https://github.com/ray-project/ray/pull/23381)