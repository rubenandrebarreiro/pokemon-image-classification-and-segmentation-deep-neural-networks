	2=a??԰@2=a??԰@!2=a??԰@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC2=a??԰@{???I??@1cAaP?iK@A?/??ѳ@I???J @rEagerKernelExecute 0*	???F?KNA2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??CR??@!?Օ???X@)??CR??@1?Օ???X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch#??fF???!"?b?&E?)#??fF???1"?b?&E?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism*S?A?Ѣ?!rv]??TN?)bX9?Ȇ?1??j?X\2?:Preprocessing2F
Iterator::ModelG=D?;???!`ߦBZQ?))??qh?1?Q%????:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMapY?????@!Z?????X@)?f???e?1c? {.e?:Preprocessing2Y
"Iterator::Model::PrivateThreadPool?q6??!D?T?P?)??UJ??b?1N??`??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?契?X@Q9???[??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{???I??@{???I??@!{???I??@      ??!       "	cAaP?iK@cAaP?iK@!cAaP?iK@*      ??!       2	?/??ѳ@?/??ѳ@!?/??ѳ@:	???J @???J @!???J @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?契?X@y9???[??