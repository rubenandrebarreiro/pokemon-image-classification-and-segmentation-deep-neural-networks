	_Pv?@_Pv?@!_Pv?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC_Pv?@?(?m6?@1k*??.hL@Ad?]K???I\??@rEagerKernelExecute 0*	O??n?H@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetchd?w?W??!??x`??L@)d?w?W??1??x`??L@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism]p????!b.??V=U@)?b*?????1X?~yE>;@:Preprocessing2F
Iterator::Model?????k??!      Y@)??hUMp?1>?U @:Preprocessing2Y
"Iterator::Model::PrivateThreadPool???Z?a??!?E?\??V@)?ص?ݒl?1[y!??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI????X@Q?
??U??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(?m6?@?(?m6?@!?(?m6?@      ??!       "	k*??.hL@k*??.hL@!k*??.hL@*      ??!       2	d?]K???d?]K???!d?]K???:	\??@\??@!\??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????X@y?
??U??