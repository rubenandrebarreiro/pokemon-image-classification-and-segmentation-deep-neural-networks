	}i???@}i???@!}i???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC}i???@Oϻ??w?@1'??Q?)K@A?9??q@@I?+?j?s@rEagerKernelExecute 0*	?/??PA2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator
??$???@!?????X@)
??$???@1?????X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch??j??!"?q7??D?)??j??1"?q7??D?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelismM??y ???!(?#VXN?)?p????1`?ؽ<3?:Preprocessing2Y
"Iterator::Model::PrivateThreadPoolDM??(#??!ڡ??YP?)s???i?1k?a?I??:Preprocessing2F
Iterator::ModelE?u?????!/?8??Q?)?HP?h?1E?d?(t?:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap?k?˞??@!????X@)?Ϲ???d?1??J???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?յ???X@Q????Q??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Oϻ??w?@Oϻ??w?@!Oϻ??w?@      ??!       "	'??Q?)K@'??Q?)K@!'??Q?)K@*      ??!       2	?9??q@@?9??q@@!?9??q@@:	?+?j?s@?+?j?s@!?+?j?s@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?յ???X@y????Q??