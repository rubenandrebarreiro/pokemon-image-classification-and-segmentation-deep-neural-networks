	?,?=ϳ@?,?=ϳ@!?,?=ϳ@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?,?=ϳ@?#????@1X?QoH@Ah??s???Ic???n@rEagerKernelExecute 0*	*\???XM@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::PrefetchCV?zN??!?;?b?E@)CV?zN??1?;?b?E@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism??ݰmQ??!?Mg-??R@)??d?`T??1?%{#>@:Preprocessing2F
Iterator::Model???bE??!      Y@)o??m?~?1J?\?)@:Preprocessing2Y
"Iterator::Model::PrivateThreadPool????-??!??<?U@)???O?~?1???8?)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIRT?X@QQ?V????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?#????@?#????@!?#????@      ??!       "	X?QoH@X?QoH@!X?QoH@*      ??!       2	h??s???h??s???!h??s???:	c???n@c???n@!c???n@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qRT?X@yQ?V????