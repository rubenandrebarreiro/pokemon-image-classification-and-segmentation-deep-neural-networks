	???HG?@???HG?@!???HG?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???HG?@?E{??	?@1?_u?HIL@A?\?C????I^=?1?@rEagerKernelExecute 0*	X9??VI@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch??RAE՟?!z˼3ͫN@)??RAE՟?1z˼3ͫN@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism???M~???!?_?ѸU@)8???n???1?藌??9@:Preprocessing2F
Iterator::Model@?@?w???!      Y@)?\?	?m?1e3?b9p@:Preprocessing2Y
"Iterator::Model::PrivateThreadPoolC??6??!???i?8W@):vP??h?1>?&ͪ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIV??N$?X@Q??X?v??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?E{??	?@?E{??	?@!?E{??	?@      ??!       "	?_u?HIL@?_u?HIL@!?_u?HIL@*      ??!       2	?\?C?????\?C????!?\?C????:	^=?1?@^=?1?@!^=?1?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qV??N$?X@y??X?v??