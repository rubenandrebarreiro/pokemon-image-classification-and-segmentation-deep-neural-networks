	V??6C??@V??6C??@!V??6C??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCV??6C??@]?`7`|?@1f/ۊH@A???9#J??Iog_y??@rEagerKernelExecute 0*	?p=
??F@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::PrefetchSh?
??!9?U?G@)Sh?
??19?U?G@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism?Y?Nܣ?!?5? _=U@)`??5!???1?RV???B@:Preprocessing2Y
"Iterator::Model::PrivateThreadPool?x?????!4p?c9W@)[?a/?m?1???'M?@:Preprocessing2F
Iterator::Model :̗`??!      Y@)	^?j?1??X??i@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIcԾܵ?X@QtΕ?%??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]?`7`|?@]?`7`|?@!]?`7`|?@      ??!       "	f/ۊH@f/ۊH@!f/ۊH@*      ??!       2	???9#J?????9#J??!???9#J??:	og_y??@og_y??@!og_y??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qcԾܵ?X@ytΕ?%??