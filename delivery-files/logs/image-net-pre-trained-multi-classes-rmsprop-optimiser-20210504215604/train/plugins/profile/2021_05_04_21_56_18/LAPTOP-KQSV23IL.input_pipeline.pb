	??nP?@??nP?@!??nP?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??nP?@=HO???@1*V?ܐN@A???<,Ԫ?I? OZ?(@rEagerKernelExecute 0*	8?A`?'?@2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorw?}9?J@!??s?<?X@)w?}9?J@1??s?<?X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch??5"??!?Fڠ~??)??5"??1?Fڠ~??:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap?R?Z?J@!?_?9?X@)??O??1X?@L?ğ?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelismfi??r???!ڵ???X??)?;2V????1\Y??f??:Preprocessing2F
Iterator::ModelQ??ڦx??!U??^???)?4?;?h?1">A6C w?:Preprocessing2Y
"Iterator::Model::PrivateThreadPool	?c???!t?+????)??)??f?1?	3ri,u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?^σ??X@Qx?P?/??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	=HO???@=HO???@!=HO???@      ??!       "	*V?ܐN@*V?ܐN@!*V?ܐN@*      ??!       2	???<,Ԫ????<,Ԫ?!???<,Ԫ?:	? OZ?(@? OZ?(@!? OZ?(@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?^σ??X@yx?P?/??